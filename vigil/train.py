from __future__ import annotations

# The loss and loop follow the fine-tuning notebook of upstream Laya
# (https://github.com/NandhaKishorM/laya, Apache-2.0).

import argparse
import json
import os
import random
import time

import torch
from laya.common import build_model, proper_reward
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from vigil.calibrate import apply_temperatures, fit_temperatures
from vigil.checkpoint import require_pinned_revision
from vigil.rows import BASE_MODEL_ID, HEAD_MAX_LEN, MAX_LEN

_MASK_FILL = -1e4


def collate(items: list, pad_id: int) -> dict:
    n = len(items)
    seq_len = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, seq_len), pad_id, dtype=torch.long)
    att = torch.zeros((n, seq_len), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        length, k = len(it["ids"]), len(it["markers"])
        ids[i, :length] = torch.tensor(it["ids"])
        att[i, :length] = 1
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def split_calibration(items: list, seed: int, calib_max: int):
    """Hold out whole conversations: items sharing a "conv" id stay on the same side.

    Items without a "conv" id (older item files, unit tests) each count as their own group.
    """
    groups: dict = {}
    for i, it in enumerate(items):
        groups.setdefault(it.get("conv", ("item", i)), []).append(i)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = min(calib_max, len(items) // 10)
    calib_idx: list = []
    for key in keys:
        if len(calib_idx) >= target:
            break
        calib_idx += groups[key]
    held = set(calib_idx)
    calib = [items[i] for i in sorted(held)]
    train_items = [items[i] for i in range(len(items)) if i not in held]
    return train_items, calib


def compute_loss(logits, batch, *, sigma: float, group_size: int, rl_weight: float):
    mask = batch["marker_mask"].to(logits.device)
    target = batch["target"].to(logits.device)
    qtype = batch["qtype"].to(logits.device)
    k = mask.sum(-1, keepdim=True).float()

    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask  # zero-mean over the real options
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, _MASK_FILL), -1)

    with torch.no_grad():
        reward = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = reward - reward.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)

    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    loss_rl = -(adv * logp).mean()
    log_probs = torch.log_softmax(logits.masked_fill(~mask, _MASK_FILL), -1)
    loss_ce = -(target * log_probs).sum(-1).mean()
    return loss_ce + rl_weight * loss_rl, reward.mean()


def _save(model, tok, cfg, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def _forward(model, batch, device):
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits, act = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["marker_pos"].to(device), batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        )
    return logits.float(), act


def _calibrate(model, calib_items, pad_id, device) -> list:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(calib_items), 16):
            chunk = calib_items[start:start + 16]
            logits, _ = _forward(model, collate(chunk, pad_id), device)
            arr = logits.cpu().numpy()
            for row, it in enumerate(chunk):
                preds.append((it["qtype"], arr[row, : len(it["markers"])].tolist(), it["target"]))
    return fit_temperatures(preds)


def train(model_dir, items, output_dir, *, epochs=4, micro_batch=8, grad_accum=4,
          lr_encoder=2.5e-5, lr_head=1e-4, rl_weight=1.0, group_size=4,
          sigma_start=0.4, sigma_end=0.1, calib_max=1600, max_steps=None, seed=20260922) -> dict:
    # ponytail: single-process, add DDP only if one GPU is too slow
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg = {**cfg, "gradient_checkpointing": True, "max_tokens_per_batch": 4096,
           "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN}

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    train_items, calib_items = split_calibration(items, seed, calib_max)
    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc, "lr": lr_encoder}, {"params": head, "lr": lr_head}], weight_decay=0.01)
    total_updates = max(1, (len(train_items) // (micro_batch * grad_accum)) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    losses, step, t0 = [], 0, time.time()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(train_items)
        sigma = sigma_start + (sigma_end - sigma_start) * epoch / max(1, epochs - 1)
        optimizer.zero_grad(set_to_none=True)
        for b in range(0, len(train_items), micro_batch):
            batch = collate(train_items[b:b + micro_batch], tok.pad_token_id)
            logits, act = _forward(model, batch, device)
            loss, reward = compute_loss(logits, batch, sigma=sigma, group_size=group_size,
                                        rl_weight=rl_weight)
            scaler.scale((loss + 0.0 * act.sum()) / grad_accum).backward()
            losses.append(loss.item())
            step += 1
            last = b + micro_batch >= len(train_items)
            if step % grad_accum == 0 or last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                rate = step * micro_batch / (time.time() - t0)
                print(f"epoch {epoch + 1}/{epochs} step {step} loss {loss.item():.4f} "
                      f"reward {reward.item():.3f} {rate:.1f} items/s", flush=True)
            if max_steps and step >= max_steps:
                break
        _save(model, tok, cfg, os.path.join(output_dir, "checkpoint_latest"))
        if max_steps and step >= max_steps:
            break

    temps = _calibrate(model, calib_items, tok.pad_token_id, device)
    _save(model, tok, apply_temperatures(cfg, temps), output_dir)
    return {"losses": losses, "temperatures": temps, "output_dir": output_dir}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune Laya on typed items")
    p.add_argument("--model-id", default=BASE_MODEL_ID)
    p.add_argument("--base-revision", default=None,
                   help="commit sha (or tag) of --model-id; required, 'main' is refused")
    p.add_argument("--items", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--rl-weight", type=float, default=1.0)
    p.add_argument("--calib-max", type=int, default=1600)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260922, help="shuffling, calibration split and RLCD noise")
    return p.parse_args(argv)


def main() -> None:
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config

    a = parse_args()
    base_revision = require_pinned_revision(a.base_revision, f"base model {a.model_id!r}")
    model_dir = snapshot_download(a.model_id, revision=base_revision)
    _fix_tokenizer_config(model_dir)
    items = torch.load(a.items, weights_only=True)
    result = train(model_dir, items, a.output_dir, epochs=a.epochs, micro_batch=a.micro_batch,
                   grad_accum=a.grad_accum, rl_weight=a.rl_weight, calib_max=a.calib_max,
                   max_steps=a.max_steps, seed=a.seed)
    # Record which base commit the weights were trained from.
    os.makedirs(a.output_dir, exist_ok=True)
    with open(os.path.join(a.output_dir, "BASE_MODEL"), "w") as f:
        f.write(f"{a.model_id}@{base_revision}\n")
    print("temperatures (choice, score, noul):", [round(t, 3) for t in result["temperatures"]])


if __name__ == "__main__":
    main()

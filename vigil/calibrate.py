from __future__ import annotations

# The temperature fit follows the fine-tuning notebook of upstream Laya
# (https://github.com/NandhaKishorM/laya, Apache-2.0).

import torch
from laya.common import QTYPES

_DEFAULT_TEMPERATURE = 1.2
TEMP_MIN, TEMP_MAX = 0.5, 5.0  # the range laya's runtime honours; anything outside is silently replaced
_MIN_SAMPLES = 10


def fit_one_temp(sel: list) -> float:
    if len(sel) < _MIN_SAMPLES:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    logits = torch.full((len(sel), kmax), -1e4)
    targets = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        logits[i, : len(z)] = torch.tensor(z)
        targets[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), TEMP_MIN, TEMP_MAX).item())


def fit_temperatures(preds: list) -> list:
    temps = [_DEFAULT_TEMPERATURE] * len(QTYPES)
    for qtype in QTYPES.values():
        sel = [(z, t) for q, z, t in preds if q == qtype]
        if sel:
            temps[qtype] = fit_one_temp(sel)
    return temps


def apply_temperatures(cfg: dict, temps: list) -> dict:
    out = {k: v for k, v in cfg.items() if k != "temperature_by_options"}
    out["temperature"] = list(temps)
    out["fine_tuned"] = True
    return out

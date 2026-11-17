"""Pinning and validation helpers for Hub base models and laya checkpoints.

Stdlib only, so the checks run without torch or laya installed.
"""

from __future__ import annotations

from pathlib import Path

# Files that unpickle or run as Python are refused anywhere in a checkpoint.
UNSAFE_SUFFIXES = frozenset(
    {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".py", ".pyc"}
)
# Inside encoder/ and tokenizer/, which HF parses, only these suffixes are allowed.
ALLOWED_SUFFIXES = frozenset({".json", ".safetensors", ".txt", ".model", ".jinja"})
REQUIRED_FILES = ("rl_agent_config.json", "model.safetensors")
CHECKPOINT_SUBDIRS = ("encoder", "tokenizer")


def require_pinned_revision(revision: str | None, what: str) -> str:
    """Return an explicit revision or fail; 'main' and empty values are refused."""
    if not revision or revision == "main":
        raise ValueError(
            f"{what} needs an explicit pinned revision (commit sha or tag); "
            f"got {revision!r}. 'main' is not accepted."
        )
    return revision


def validate_checkpoint_dir(path: str | Path) -> None:
    """Refuse a local checkpoint directory that could execute or parse unsafe files.

    laya opens the required root files plus encoder/ and tokenizer/, which must
    exist so laya never fetches an encoder or tokenizer by Hub name. Other root
    files (BASE_MODEL, exported .onnx, litert outputs) are not opened by laya and
    are only checked against UNSAFE_SUFFIXES.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"checkpoint {root} is not a local directory")

    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    missing += [name for name in CHECKPOINT_SUBDIRS if not (root / name).is_dir()]
    if missing:
        raise ValueError(f"checkpoint {root} is missing {', '.join(missing)}")

    for entry in sorted(root.rglob("*")):
        if entry.is_file() and entry.suffix in UNSAFE_SUFFIXES:
            raise ValueError(
                f"unsafe file in checkpoint: {entry.relative_to(root)} "
                "(pickle and Python files are refused)"
            )

    for name in CHECKPOINT_SUBDIRS:
        for entry in sorted((root / name).rglob("*")):
            if entry.is_file() and entry.suffix not in ALLOWED_SUFFIXES:
                raise ValueError(
                    f"unexpected file in checkpoint: {entry.relative_to(root)} "
                    f"(allowed in {name}/: {', '.join(sorted(ALLOWED_SUFFIXES))})"
                )

"""Checkpoint guard and revision pin checks. Stdlib only, no torch or laya needed."""

import pytest

from vigil.checkpoint import require_pinned_revision, validate_checkpoint_dir

SHA = "c" * 40


def _make_checkpoint(root):
    """Minimal layout that vigil.train writes."""
    (root / "encoder").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    (root / "rl_agent_config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"")
    (root / "encoder" / "config.json").write_text("{}")
    (root / "tokenizer" / "tokenizer.json").write_text("{}")
    (root / "tokenizer" / "chat_template.jinja").write_text("{{ messages }}")
    return root


def _add_pipeline_outputs(root):
    """Files that later pipeline steps write into the same checkpoint directory."""
    (root / "BASE_MODEL").write_text("org/base@" + SHA + "\n")
    (root / "vigil.onnx").write_bytes(b"onnx")
    (root / "litert").mkdir()
    (root / "litert" / "horizon.litertlm").write_bytes(b"litert")
    return root


def test_require_pinned_revision_accepts_explicit_pin():
    """A commit sha passes through unchanged."""
    assert require_pinned_revision(SHA, "base") == SHA


@pytest.mark.parametrize("bad", [None, "", "main"])
def test_require_pinned_revision_rejects_missing_or_main(bad):
    """Missing revisions and the mutable 'main' branch are refused."""
    with pytest.raises(ValueError, match="pinned revision"):
        require_pinned_revision(bad, "base")


def test_validate_accepts_minimal_layout(tmp_path):
    """The layout vigil.train writes passes validation."""
    validate_checkpoint_dir(_make_checkpoint(tmp_path / "ckpt"))


def test_validate_accepts_checkpoint_after_export_steps(tmp_path):
    """BASE_MODEL, exported ONNX and LiteRT outputs do not block re-evaluation."""
    root = _add_pipeline_outputs(_make_checkpoint(tmp_path / "ckpt"))

    validate_checkpoint_dir(root)


@pytest.mark.parametrize(
    "name", ["pytorch_model.bin", "tokenizer/extra.pkl", "encoder/x.pt"]
)
def test_validate_rejects_pickle_files_anywhere(tmp_path, name):
    """Pickle-based files are refused, at the root or inside the subtrees."""
    root = _make_checkpoint(tmp_path / "ckpt")
    (root / name).write_bytes(b"\x80")

    with pytest.raises(ValueError, match="unsafe file"):
        validate_checkpoint_dir(root)


def test_validate_rejects_python_files(tmp_path):
    """Python files that could run as remote code are refused."""
    root = _make_checkpoint(tmp_path / "ckpt")
    (root / "encoder" / "modeling_custom.py").write_text("print('x')")

    with pytest.raises(ValueError, match="unsafe file"):
        validate_checkpoint_dir(root)


def test_validate_rejects_unexpected_file_in_subtree(tmp_path):
    """HF parses encoder/ and tokenizer/, so only allow-listed suffixes are accepted there."""
    root = _make_checkpoint(tmp_path / "ckpt")
    (root / "tokenizer" / "notes.md").write_text("x")

    with pytest.raises(ValueError, match="unexpected file"):
        validate_checkpoint_dir(root)


def test_validate_rejects_missing_encoder(tmp_path):
    """A missing encoder/ would make laya fetch the encoder by Hub name, so refuse."""
    root = _make_checkpoint(tmp_path / "ckpt")
    (root / "encoder" / "config.json").unlink()
    (root / "encoder").rmdir()

    with pytest.raises(ValueError, match="missing encoder"):
        validate_checkpoint_dir(root)


def test_validate_rejects_non_directory(tmp_path):
    """A Hub id or missing path is not a local checkpoint."""
    with pytest.raises(ValueError, match="not a local directory"):
        validate_checkpoint_dir(tmp_path / "convaiinnovations-laya")

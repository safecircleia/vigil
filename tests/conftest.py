import os

import pytest

BASE_ID = "convaiinnovations/laya-multilingual"
# Tests that download the base model must use a pinned commit, never 'main'.
BASE_REVISION = os.environ.get("VIGIL_BASE_REVISION", "")


@pytest.fixture(scope="session")
def base_dir():
    if not BASE_REVISION or BASE_REVISION == "main":
        pytest.skip("set VIGIL_BASE_REVISION to a pinned commit sha to fetch the base model")
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(BASE_ID, revision=BASE_REVISION)
    except Exception as exc:  # offline, gated or renamed
        pytest.skip(f"cannot fetch {BASE_ID}@{BASE_REVISION}: {exc}")


@pytest.fixture(scope="session")
def tok(base_dir):
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    _fix_tokenizer_config(base_dir)
    return AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))

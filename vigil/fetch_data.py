from __future__ import annotations

import argparse
import json
import os

DEFAULT_REPO = "safecircleai/horizon-training-data"


def fetch(out_dir: str, revision: str, repo_id: str = DEFAULT_REPO) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    listed = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for split in ("train", "eval"):
        names = sorted(f for f in listed if f.startswith(f"processed/{split}-") and f.endswith(".parquet"))
        if not names:
            raise FileNotFoundError(f"no processed/{split}-*.parquet in {repo_id}@{revision}")
        files = [hf_hub_download(repo_id, n, repo_type="dataset", revision=revision) for n in names]
        n = 0
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for path in files:
                for messages in pq.read_table(path, columns=["messages"]).column("messages").to_pylist():
                    f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    n += 1
        counts[split] = n
    with open(os.path.join(out_dir, "REVISION"), "w") as f:
        f.write(f"{repo_id}@{revision}\n")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="Download a pinned revision of the source dataset")
    p.add_argument("--revision", required=True, help="commit sha or tag; 'main' is not accepted")
    p.add_argument("--out-dir", default="data/source")
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    a = p.parse_args()
    if a.revision == "main":
        raise SystemExit("pin an explicit revision (commit sha or tag), not 'main'")
    print(fetch(a.out_dir, a.revision, a.repo_id))


if __name__ == "__main__":
    main()

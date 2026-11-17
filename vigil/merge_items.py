from __future__ import annotations

import argparse

import torch


def merge(paths: list, output: str) -> int:
    merged, seen = [], set()
    for path in paths:
        items = torch.load(path, weights_only=True)
        convs = {it["conv"] for it in items}
        clash = seen & convs
        if clash:
            raise ValueError(f"conversation ids collide across files (e.g. {sorted(clash)[:3]}): use distinct --source-tag values")
        seen |= convs
        merged.extend(items)
    torch.save(merged, output)
    return len(merged)


def main() -> None:
    p = argparse.ArgumentParser(description="Concatenate item files from different sources")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(f"merged {merge(a.inputs, a.output)} items into {a.output}")


if __name__ == "__main__":
    main()

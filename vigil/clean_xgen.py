"""Drop generated conversations that contain a content-less turn (an artifact of the old parser).

Rewrites each file in place and keeps the original next to it as <file>.orig (never overwritten).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from vigil.rows import split_messages


def has_empty_turn(state: str) -> bool:
    return any(":" in line and not line.split(":", 1)[1].strip("*_ ") for line in state.split("\n"))


def clean_file(path: str) -> dict:
    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    kept, dropped, out = 0, 0, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if has_empty_turn(split_messages(json.loads(line)["messages"])[0]):
                dropped += 1
            else:
                kept += 1
                out.append(line if line.endswith("\n") else line + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)
    return {"kept": kept, "dropped": dropped}


def main() -> None:
    p = argparse.ArgumentParser(description="Remove generated rows that have an empty turn")
    p.add_argument("files", nargs="+")
    for path in p.parse_args().files:
        print(path, clean_file(path))


if __name__ == "__main__":
    main()

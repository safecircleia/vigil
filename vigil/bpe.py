"""A small BPE tokenizer for mmBERT's (Gemma-2 style) vocabulary, and the compact file the Android app loads.

mmBERT's tokenizer.json is 32 MB of JSON. At run time only three things are needed: the id of every
single-character token, the merge table, and the handful of "added" tokens that are matched literally
in the raw text. `write_compact` stores exactly that; `Bpe` reads it back and tokenizes the way
`tokenizers` does for this vocabulary (space -> "▁", a "▁" prepended, split before each "▁", byte
fallback, rank-ordered merges). The Kotlin tokenizer in the app is a line-by-line port of `Bpe`, and
`scripts/export_ondevice_assets.py` checks `Bpe` against the Hugging Face tokenizer before shipping.

File layout (little endian):
  "VTOK", int32 n_chars, int32 n_byte_tokens(=256), int32 n_added, int32 n_merges, int32 unk_id
  n_chars   x (int32 codepoint, int32 id)
  256       x  int32 id of <0xXX>, indexed by byte value (-1 if absent)
  n_added   x (int32 id, uint16 n_bytes, utf-8 bytes), longest first
  n_merges  x (int64 key = left << 32 | right, int32 rank, int32 result), sorted by key
"""
from __future__ import annotations

import bisect
import json
import struct

MAGIC = b"VTOK"
SPACE = "▁"  # "▁"


def write_compact(tokenizer_json: str, out_path: str) -> dict:
    with open(tokenizer_json, encoding="utf-8") as f:
        t = json.load(f)
    model = t["model"]
    assert model["type"] == "BPE" and model["byte_fallback"] and not model["ignore_merges"]
    vocab = model["vocab"]
    chars = sorted((ord(tok), i) for tok, i in vocab.items() if len(tok) == 1)
    byte_ids = [vocab.get("<0x%02X>" % b, -1) for b in range(256)]
    # special tokens too: tokenizers matches "<eos>" typed into the text as the special token, and so must we
    added = sorted(((a["content"], a["id"]) for a in t["added_tokens"]), key=lambda x: (-len(x[0]), x[0]))
    merges = []
    for rank, pair in enumerate(model["merges"]):
        left, right = pair if isinstance(pair, list) else pair.split(" ")
        merges.append(((vocab[left] << 32) | vocab[right], rank, vocab[left + right]))
    merges.sort()
    out = bytearray(MAGIC)
    out += struct.pack("<5i", len(chars), 256, len(added), len(merges), vocab[model["unk_token"]])
    for cp, i in chars:
        out += struct.pack("<2i", cp, i)
    out += struct.pack("<256i", *byte_ids)
    for content, i in added:
        raw = content.encode("utf-8")
        out += struct.pack("<iH", i, len(raw)) + raw
    for key, rank, result in merges:
        out += struct.pack("<qii", key, rank, result)
    with open(out_path, "wb") as f:
        f.write(out)
    return {"bytes": len(out), "chars": len(chars), "added": len(added), "merges": len(merges)}


class Bpe:
    def __init__(self, path: str):
        data = open(path, "rb").read()
        assert data[:4] == MAGIC
        n_chars, n_bytes, n_added, n_merges, self.unk = struct.unpack_from("<5i", data, 4)
        p = 24
        self.char_id = {}
        for _ in range(n_chars):
            cp, i = struct.unpack_from("<2i", data, p)
            self.char_id[cp] = i
            p += 8
        self.byte_id = list(struct.unpack_from("<256i", data, p))
        p += 1024
        self.added = []
        for _ in range(n_added):
            i, n = struct.unpack_from("<iH", data, p)
            self.added.append((data[p + 6: p + 6 + n].decode("utf-8"), i))
            p += 6 + n
        self.keys, self.rank, self.result = [], [], []
        for _ in range(n_merges):
            key, rank, result = struct.unpack_from("<qii", data, p)
            self.keys.append(key)
            self.rank.append(rank)
            self.result.append(result)
            p += 16

    def _merge(self, left: int, right: int):
        key = (left << 32) | right
        i = bisect.bisect_left(self.keys, key)
        if i < len(self.keys) and self.keys[i] == key:
            return self.rank[i], self.result[i]
        return None

    def _piece(self, piece: str) -> list:
        symbols = []
        for ch in piece:
            i = self.char_id.get(ord(ch))
            if i is not None:
                symbols.append(i)
                continue
            ids = [self.byte_id[b] for b in ch.encode("utf-8")]
            if min(ids) < 0:
                if not symbols or symbols[-1] != self.unk:
                    symbols.append(self.unk)  # unknown characters fuse into one <unk>
            else:
                symbols.extend(ids)
        while len(symbols) > 1:
            best = None
            for k in range(len(symbols) - 1):
                m = self._merge(symbols[k], symbols[k + 1])
                if m is not None and (best is None or m[0] < best[1]):
                    best = (k, m[0], m[1])
            if best is None:
                break
            k, _, result = best
            symbols[k: k + 2] = [result]
        return symbols

    def _segment(self, text: str) -> list:
        text = text.replace(" ", SPACE)
        if not text:
            return []
        if not text.startswith(SPACE):
            text = SPACE + text
        pieces, start = [], 0
        for i in range(1, len(text)):
            if text[i] == SPACE:
                pieces.append(text[start:i])
                start = i
        pieces.append(text[start:])
        ids = []
        for piece in pieces:
            ids.extend(self._piece(piece))
        return ids

    def encode(self, text: str) -> list:
        """Token ids without special tokens, like `tok(text, add_special_tokens=False)["input_ids"]`."""
        ids, start, i = [], 0, 0
        while i < len(text):
            hit = next(((c, tid) for c, tid in self.added if text.startswith(c, i)), None)
            if hit is None:
                i += 1
                continue
            ids.extend(self._segment(text[start:i]))
            ids.append(hit[1])
            i += len(hit[0])
            start = i
        ids.extend(self._segment(text[start:]))
        return ids

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
import difflib


SENTENCE_KEYS = ("sentence", "korean_sentence", "korean_text", "korean text")
GLOSS_KEYS = ("gloss", "gloss_seq", "descriptor")


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(data, dict):
        for item in data.values():
            if isinstance(item, dict):
                yield item


def _get_sentence(rec: dict[str, Any]) -> str | None:
    for key in SENTENCE_KEYS:
        val = rec.get(key)
        if val is not None:
            s = str(val).strip()
            if s:
                return s
    return None


def _get_gloss_tokens(rec: dict[str, Any]) -> list[str]:
    gloss = None
    for key in GLOSS_KEYS:
        gloss = rec.get(key)
        if gloss is not None:
            break

    if gloss is None:
        return []
    if isinstance(gloss, list):
        return [str(t).strip() for t in gloss if str(t).strip()]
    if isinstance(gloss, str):
        return [t for t in gloss.strip().split() if t]
    return [str(gloss).strip()] if str(gloss).strip() else []


def _build_index(path: Path) -> tuple[dict[str, list[str]], int]:
    index: dict[str, list[str]] = {}
    dup_count = 0
    for rec in _iter_records(path):
        sentence = _get_sentence(rec)
        if not sentence:
            continue
        if sentence in index:
            dup_count += 1
            continue
        index[sentence] = _get_gloss_tokens(rec)
    return index, dup_count


def _diff_tokens(a: list[str], b: list[str]) -> str:
    parts = []
    for tok in difflib.ndiff(a, b):
        if tok.startswith("- "):
            parts.append(f"-{tok[2:]}")
        elif tok.startswith("+ "):
            parts.append(f"+{tok[2:]}")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare gloss sequences between two JSON/JSONL files.")
    parser.add_argument("a", type=Path, help="First JSON/JSONL file")
    parser.add_argument("b", type=Path, help="Second JSON/JSONL file")
    parser.add_argument("--show-missing", action="store_true", help="Show sentences missing on either side")
    args = parser.parse_args()

    idx_a, dup_a = _build_index(args.a)
    idx_b, dup_b = _build_index(args.b)

    sentences_a = set(idx_a)
    sentences_b = set(idx_b)
    shared = sorted(sentences_a & sentences_b)
    only_a = sorted(sentences_a - sentences_b)
    only_b = sorted(sentences_b - sentences_a)

    same = 0
    diff = 0

    for sentence in shared:
        gloss_a = idx_a[sentence]
        gloss_b = idx_b[sentence]
        if gloss_a == gloss_b:
            same += 1
            continue
        diff += 1
        diff_str = _diff_tokens(gloss_a, gloss_b)
        print(f"Sentence: {sentence}")
        print(f"A: {' '.join(gloss_a)}")
        print(f"B: {' '.join(gloss_b)}")
        print(f"Diff: {diff_str if diff_str else '(token order differs)'}")
        print("-" * 40)

    if args.show_missing:
        for sentence in only_a:
            print(f"Missing in B: {sentence}")
        for sentence in only_b:
            print(f"Missing in A: {sentence}")

    print(
        "\nTally\n"
        f"Total A sentences: {len(sentences_a)}\n"
        f"Total B sentences: {len(sentences_b)}\n"
        f"Shared sentences: {len(shared)}\n"
        f"Gloss same: {same}\n"
        f"Gloss different: {diff}\n"
        f"Only in A: {len(only_a)}\n"
        f"Only in B: {len(only_b)}\n"
        f"Duplicate sentences skipped (A): {dup_a}\n"
        f"Duplicate sentences skipped (B): {dup_b}\n"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

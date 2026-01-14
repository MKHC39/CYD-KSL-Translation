import json
import random
from pathlib import Path


def main(json_path: Path, n: int) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)  # { "entry_id": ["sentence", "gloss_seq"], ... }

    items = list(data.items())

    if n > len(items):
        print(f"Requested {n} entries, but only {len(items)} available. Showing all.")
        n = len(items)

    for entry_id, pair in random.sample(items, n):
        sentence, gloss_seq = pair[0], pair[1]
        print(f"entry_id: {entry_id}")
        print(f"sentence: {sentence}")
        print(f"seq: {gloss_seq}")
        print("-" * 50)


if __name__ == "__main__":
    json_path = Path(__file__).resolve().parent / "ksl_sentence_gloss.json"

    n = int(input("How many random entries to print? "))
    main(json_path, n)

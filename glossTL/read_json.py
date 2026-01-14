import json
import random
from pull_script import wsl_path
from pathlib import Path


def main(jsonl_path: Path, n: int) -> None:
    sent_str = ''
    gloss_str = ''
    sent_no = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    if n > len(lines):
        print(f"Requested {n} entries, but only {len(lines)} available. Showing all.")
        n = len(lines)

    print("글로스 시퀀스\t한국어 문장")
    for line in random.sample(lines, n):
        obj = json.loads(line)

        gloss = str(obj.get("gloss", "")).replace("\t", " ").replace("\n", " ").strip()
        sent = str(obj.get("sentence", "")).replace("\t", " ").replace("\n", " ").strip()

        print(f"{gloss}\t{sent}")

        sent_no += 1
        sent_str += str(sent_no) + '. ' + obj['sentence'] + '\n' + '\n'
        gloss_str += str(sent_no) + '. ' + obj['gloss'] + '\n' + '\n'

    print(sent_str)
    print(gloss_str)


if __name__ == "__main__":
    json_path = wsl_path(r"C:\Workplace\workspaces\ksl\glossTL") / "ksl_sentence_gloss.json"

    n = int(input("How many random entries to print? "))
    main(json_path, n)

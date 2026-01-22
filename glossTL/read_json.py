import json
import random
from glossTL.pull_script import wsl_path
from pathlib import Path
import pandas as pd
from pathmagic import smart_path as sp


def main(jsonl_path: Path, n: int | None = None) -> None | pd.DataFrame:
    sent_str = ''
    gloss_str = ''
    sent_no = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]


    if n is not None:
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

    else:
        out = pd.DataFrame(columns=["gloss", "sentence"])
        for i in range(n if n else len(lines)):
            line = lines[i]
            gloss = str(json.loads(line).get("gloss", "")).replace("\t", " ").replace("\n", " ").strip()
            sentence = str(json.loads(line).get("sentence", "")).replace("\t", " ").replace("\n", " ").strip()
            out.loc[len(out)] = [gloss, sentence]
        return out


def line_lookup(jsonl_path: Path, n: int) -> None:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == n:
                obj = json.loads(line)
                gloss = str(obj.get("gloss", "")).replace("\t", " ").replace("\n", " ").strip()
                sent = str(obj.get("sentence", "")).replace("\t", " ").replace("\n", " ").strip()
                print(f"{gloss}\t{sent}")
                return



if __name__ == "__main__":
    json_path = sp(r"~/Workplace/workspaces/ksl/glossTL") / "training_sentence_gloss.jsonl"

    n = int(input("How many random entries to print? "))
    main(json_path, n)

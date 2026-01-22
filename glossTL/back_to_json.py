import json
from pathlib import Path

import pandas as pd


def csv_to_jsonl_fixed(in_csv: Path, out_jsonl: Path, pred_col: int) -> None:
    """
    For your specific CSV:
      col 1 = entry id
      col 2 = gloss
      col 3 = reference sentence
      col 4 = GPT 5.2 output
      col 5 = GPT-oss output
      col 6 = Gemini 3 Flash output
      col 7 = Gemini 3 Pro output
    pred_col: choose 4 or 7 depending on which model you want as "pred"
    """
    df = pd.read_csv(in_csv, header=None)

    # Drop completely empty rows
    df = df.dropna(how="all")

    # Drop the internal header row where gloss cell is "글로스 시퀀스"
    df = df[df[2].astype(str).str.strip() != "글로스 시퀀스"]

    # Keep only rows that have both gloss and reference sentence
    df = df.dropna(subset=[2, 3])

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            entry_id = int(row[1])
            gloss = str(row[2]).replace("\n", " ").replace("\t", " ").strip()
            sent = str(row[3]).replace("\n", " ").replace("\t", " ").strip()
            pred = "" if pd.isna(row.get(pred_col)) else str(row[pred_col]).replace("\n", " ").replace("\t", " ").strip()

            obj = {"entry_id": entry_id, "gloss": gloss, "sentence": sent, "pred": pred}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote JSONL lines: {written}")
    print(f"Saved to: {out_jsonl}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    in_csv = here / "data.csv"
    out_jsonl = here / "GeminiP-data.jsonl"

    # pred_col=4 for GPT 5.2, pred_col=7 for Gemini 3 Pro
    csv_to_jsonl_fixed(in_csv, out_jsonl, pred_col=7)

import json
from pathlib import Path

import torch
from bert_score import score


def load_preds_refs(jsonl_path: Path, pred_key: str = "pred", ref_key: str = "sentence"):
    """
    input: {"entry_id": 123, "gloss": "...", "sentence": "REF ...", "pred": "LLM OUTPUT ..."}
    """


    preds, refs = [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            pred = str(obj.get(pred_key, "")).strip()
            ref = str(obj.get(ref_key, "")).strip()
            if not pred or not ref:
                continue

            preds.append(pred.replace("\n", " "))
            refs.append(ref.replace("\n", " "))

    return preds, refs


def main(jsonl_path: Path, model_type: str = "xlm-roberta-large", batch_size: int = 32):
    preds, refs = load_preds_refs(jsonl_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F1 = score(
        preds,
        refs,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        verbose=True,
    )

    f1_list = [float(x) for x in F1]
    mean_f1 = sum(f1_list) / len(f1_list) if f1_list else float("nan")

    print(f"Examples scored: {len(f1_list)}")
    print(f"Mean BERTScore F1: {mean_f1:.6f}")
    return mean_f1


if __name__ == "__main__":
    datalist = []
    for i in ['GPT5.2-data.jsonl','GPT-oss-data.jsonl','GeminiF-data.jsonl','GeminiP-data.jsonl']:
        jsonl_path = Path(__file__).resolve().parent / i
        data = main(jsonl_path)
        datalist.append(data)
    print(['5.2', 'oss', 'F', 'P'])
    print(datalist)
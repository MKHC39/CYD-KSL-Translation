import json
from pathlib import Path

from comet import download_model, load_from_checkpoint


def load_triples(jsonl_path: Path):
    srcs, mts, refs = [], [], []
    kept = 0
    skipped = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            src = str(obj.get("gloss", "")).strip()
            mt = str(obj.get("pred", "")).strip()
            ref = str(obj.get("sentence", "")).strip()

            if not src or not mt or not ref:
                skipped += 1
                continue

            # keep it one-line to avoid weird formatting effects
            srcs.append(src.replace("\n", " ").replace("\t", " "))
            mts.append(mt.replace("\n", " ").replace("\t", " "))
            refs.append(ref.replace("\n", " ").replace("\t", " "))
            kept += 1

    return srcs, mts, refs, kept, skipped


def main(
    jsonl_path: Path,
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int = 1,
):
    srcs, mts, refs, kept, skipped = load_triples(jsonl_path)
    print(f"Loaded: {kept} triples (skipped {skipped})")

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)]
    out = model.predict(data, batch_size=batch_size, gpus=gpus)

    # out.scores is per-segment; out.system_score is the mean
    scores = out.scores
    mean_score = float(out.system_score)

    print(f"Examples scored: {len(scores)}")
    print(f"Mean COMET: {mean_score:.6f}")

    return mean_score


if __name__ == "__main__":
    datalist = []
    for i in ['GPT5.2-data.jsonl', 'GPT-oss-data.jsonl', 'GeminiF-data.jsonl', 'GeminiP-data.jsonl']:
        jsonl_path = Path(__file__).resolve().parent / i
        data = main(jsonl_path)
        datalist.append(data)
    print(['5.2', 'oss', 'F', 'P'])
    print(datalist)

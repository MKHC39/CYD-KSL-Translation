import json
from pathlib import Path

def ksl_gt_stm(
    manifest_path: str,
    out_stm_path: str,
    signer: str = "0",
    check_exists: bool = True,
):
    manifest_path = Path(manifest_path)
    out_stm_path = Path(out_stm_path)
    out_stm_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    skipped_missing = 0

    with manifest_path.open("r", encoding="utf-8") as fin, \
         out_stm_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)

            stem = row["stem"]          # fileid CorrNet uses
            label = row["label"]        # reference token string
            npz_path = Path(row["path"]) if "path" in row else None

            if check_exists and npz_path is not None and not npz_path.exists():
                skipped_missing += 1
                continue

            fout.write(f"{stem} 1 {signer} 0.0 1.79769e+308 {label}\n")
            n += 1

    print(f"[OK] wrote {n} lines -> {out_stm_path} (skipped_missing={skipped_missing})")

if __name__ == "__main__":
    ksl_gt_stm(manifest_path=r"C:\Users\CHOI\Downloads\KSL Word DataSet\cached_npz\manifest.jsonl",
    out_stm_path=r"C:\Users\CHOI\PycharmProjects\CorrNet_Plus\CorrNet_Plus_CSLR\preprocess\KSL\KSL-groundtruth-dev.stm")

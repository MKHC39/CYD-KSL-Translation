from pathlib import Path
import torch

from pydata import iter_samples
from preprocess.preprocess_clip import preprocess_stem

CACHE_DIR = Path(r"C:\Users\CHOI\Downloads\KSL Word DataSet\cached_clips")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

W_START = 1501
W_END = 1520
ANGLES = ["D", "F", "L", "R", "U"]
STEP = 2
SHIFT_MARGIN_PX = 40
IMG_W = 1920
IMG_H = 1080

def main():
    n_ok = 0
    n_fail = 0

    for sample in iter_samples(W_START, W_END, ANGLES):
        out = CACHE_DIR / f"{sample.stem}.pt"
        if out.exists():
            continue

        try:
            clip, kept_indices, meta = preprocess_stem(
                sample.stem,
                step=STEP,
                margin_px=SHIFT_MARGIN_PX,
                img_w=IMG_W,
                img_h=IMG_H,
                normalise_imagenet=True,
            )
            # save clip + label id key (w) + lengths
            payload = {
                "clip": clip.cpu(),                 # (T,3,224,224)
                "w": sample.w,                      # class identity
                "length": int(clip.shape[0]),
                "kept_indices": kept_indices,
                "meta": meta,
            }
            torch.save(payload, out)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {sample.stem}: {e}")

    print(f"done. cached={n_ok} failed={n_fail} -> {CACHE_DIR}")

if __name__ == "__main__":
    main()

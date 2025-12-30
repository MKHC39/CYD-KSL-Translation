from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

# Your existing entrypoint
from preprocess.preprocess_clip import preprocess_stem


DEFAULT_ANGLES: Tuple[str, ...] = ("D", "F", "L", "R", "U")
P_LIST = list(range(1, 17))

@dataclass(frozen=True)
class CacheItem:
    stem: str
    w: int
    angle: str


def iter_stems(w_start: int, w_end: int, p_list: Sequence[int], angles: Sequence[str]) -> Iterable[CacheItem]:
    for p in p_list:
        for w in range(w_start, w_end + 1):
            for a in angles:
                stem = f"NIA_SL_WORD{w:04d}_REAL{p:02d}_{a}"
                yield CacheItem(stem=stem, w=w, angle=a)



def clip_chw01_to_uint8_thwc(clip: torch.Tensor) -> np.ndarray:
    """
    clip: (T,3,H,W) float32 in [0,1]  (your preprocess_stem when normalise_imagenet=False)
    returns: (T,H,W,3) uint8 in [0,255]
    """
    if clip.ndim != 4 or clip.shape[1] != 3:
        raise ValueError(f"Expected (T,3,H,W), got {tuple(clip.shape)}")
    # (T,3,H,W) -> (T,H,W,3)
    thwc = clip.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    thwc = np.clip(np.rint(thwc * 255.0), 0, 255).astype(np.uint8)
    return thwc


def save_npz(
    out_path: Path,
    video_uint8_thwc: np.ndarray,
    label_id: int,
    label_str: str,
    kept_indices: List[int],
    meta: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Store meta as JSON text (avoids np pickle objects in the clip cache)
    meta_json = json.dumps(meta, ensure_ascii=False)

    np.savez_compressed(
        out_path,
        video=video_uint8_thwc,                 # (T,H,W,3) uint8
        label_id=np.int32(label_id),
        label_str=np.array(label_str),
        kept_indices=np.asarray(kept_indices, dtype=np.int32),
        meta_json=np.array(meta_json),
    )


def build_cache(
    cache_root: Path,
    gloss_dict_out: Path,
    manifest_out: Path,
    w_start: int,
    w_end: int,
    angles: Sequence[str],
    step: int,
    margin_px: int,
    overwrite: bool = False,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)

    label_to_id: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}
    next_id = 1  # CorrNet-style ids typically start at 1 (0 is often blank in CTC)

    manifest_lines: List[str] = []

    ok = 0
    fail = 0
    skipped = 0

    for item in iter_stems(w_start, w_end, P_LIST, angles):
        out_path = cache_root / f"{item.stem}.npz"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            clip01, kept_indices, meta = preprocess_stem(
                item.stem,
                step=step,
                margin_px=margin_px,
                normalise_imagenet=False,
            )

            label_str = meta.get("label")
            if not isinstance(label_str, str) or not label_str:
                raise RuntimeError(f"Missing/invalid label in meta for {item.stem}: {label_str!r}")

            if label_str not in label_to_id:
                label_to_id[label_str] = next_id
                next_id += 1
                label_counts[label_str] = 0

            # Count “how many clips (angles) for this label”
            label_counts[label_str] += 1

            label_id = label_to_id[label_str]
            video_uint8 = clip_chw01_to_uint8_thwc(clip01)

            save_npz(
                out_path=out_path,
                video_uint8_thwc=video_uint8,
                label_id=label_id,
                label_str=label_str,
                kept_indices=kept_indices,
                meta=meta,
            )

            # Simple manifest row (jsonl)
            manifest_lines.append(
                json.dumps(
                    {
                        "stem": item.stem,
                        "w": item.w,
                        "angle": item.angle,
                        "npz": str(out_path),
                        "label": label_str,
                        "label_id": label_id,
                        "T": int(video_uint8.shape[0]),
                    },
                    ensure_ascii=False,
                )
            )

            ok += 1
            print(f"[OK] {item.stem} label={label_str} T={video_uint8.shape[0]} -> {out_path}")

        except Exception as e:
            fail += 1
            print(f"[FAIL] {item.stem}: {e}")

    # CorrNet-like gloss_dict format: {label_str: [id, count]}
    gloss_dict: Dict[str, List[int]] = {
        lab: [int(label_to_id[lab]), int(label_counts[lab])]
        for lab in sorted(label_to_id.keys())
    }

    gloss_dict_out.parent.mkdir(parents=True, exist_ok=True)
    # This produces a .npy containing an object (dict). CorrNet loads with allow_pickle=True.
    np.save(gloss_dict_out, gloss_dict)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text("\n".join(manifest_lines), encoding="utf-8")

    print(
        f"[DONE] ok={ok} fail={fail} skipped_existing={skipped} "
        f"unique_labels={len(gloss_dict)} cache_root={cache_root}"
    )
    print(f"[WROTE] gloss_dict: {gloss_dict_out}")
    print(f"[WROTE] manifest:  {manifest_out}")


if __name__ == "__main__":
    # --------- EDIT THESE ----------
    CACHE_ROOT = Path(r"C:\Users\CHOI\Downloads\KSL Word DataSet\cached_npz")
    GLOSS_DICT_OUT = Path(r"C:\Users\CHOI\Downloads\KSL Word DataSet\cached_npz\gloss_dict.npy")
    MANIFEST_OUT = Path(r"C:\Users\CHOI\Downloads\KSL Word DataSet\cached_npz\manifest.jsonl")

    W_START = 1501
    W_END = 1510
    ANGLES = DEFAULT_ANGLES

    STEP = 1
    MARGIN_PX = 40
    OVERWRITE = False
    # ------------------------------

    build_cache(
        cache_root=CACHE_ROOT,
        gloss_dict_out=GLOSS_DICT_OUT,
        manifest_out=MANIFEST_OUT,
        w_start=W_START,
        w_end=W_END,
        angles=ANGLES,
        step=STEP,
        margin_px=MARGIN_PX,
        overwrite=OVERWRITE,
    )

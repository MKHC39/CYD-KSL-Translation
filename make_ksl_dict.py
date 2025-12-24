"""
make_ksl_gloss_dict.py

Creates CorrNet-style gloss_dict.npy:
  { label_str: [class_id, count], ... }
where class_id starts at 1 (0 reserved for CTC blank).

Counts are computed by iterating stems and counting *existing* clips.

Run this in the SAME venv/NumPy version that will load the .npy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Reuse your existing parser + dataset roots so paths stay consistent.
from preprocess.preprocess_clip import MORPHEME_ROOT, VIDEO_ROOT, load_morpheme

DEFAULT_ANGLES: Tuple[str, ...] = ("D", "F", "L", "R", "U")


def iter_stems(w_start: int, w_end: int, angles: Sequence[str]) -> List[str]:
    stems: List[str] = []
    for w in range(w_start, w_end + 1):
        for a in angles:
            stems.append(f"NIA_SL_WORD{w:04d}_REAL01_{a}")
    return stems


def build_counts(
    w_start: int,
    w_end: int,
    angles: Sequence[str] = DEFAULT_ANGLES,
    require_video: bool = True,
) -> Dict[str, int]:
    """
    Returns: {label_str: count}
    count increments once per stem that exists (morpheme json present, and optionally video present).
    """
    counts: Dict[str, int] = {}
    for stem in iter_stems(w_start, w_end, angles):
        mp = MORPHEME_ROOT / f"{stem}_morpheme.json"
        if not mp.exists():
            continue

        if require_video:
            vp = VIDEO_ROOT / f"{stem}.mp4"
            if not vp.exists():
                continue

        # load_morpheme returns (start_s, end_s, label_str)
        _, _, label = load_morpheme(mp)

        counts[label] = counts.get(label, 0) + 1

    return counts


def build_gloss_dict_from_counts(counts: Dict[str, int]) -> Dict[str, List[int]]:
    """
    CorrNet format: sort by gloss string, then assign:
      gloss_dict[gloss] = [idx+1, count]
    """
    gloss_dict: Dict[str, List[int]] = {}
    for idx, gloss in enumerate(sorted(counts.keys())):
        gloss_dict[gloss] = [idx + 1, int(counts[gloss])]
    return gloss_dict


def main():
    # ---------------- EDIT THESE ----------------
    W_START = 1501
    W_END = 3000
    ANGLES = DEFAULT_ANGLES
    REQUIRE_VIDEO = True  # True = count only if mp4 exists too

    OUT_PATH = Path("preprocess") / "KSL" / "gloss_dict.npy"
    # -------------------------------------------

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    counts = build_counts(W_START, W_END, ANGLES, require_video=REQUIRE_VIDEO)
    gloss_dict = build_gloss_dict_from_counts(counts)

    np.save(str(OUT_PATH), gloss_dict, allow_pickle=True)

    print(f"[OK] saved: {OUT_PATH}")
    print(f"num_glosses: {len(gloss_dict)}")
    print(f"num_classes (with CTC blank): {len(gloss_dict) + 1}")
    # quick sanity preview
    some = list(gloss_dict.items())[:5]
    print("head:", some)


if __name__ == "__main__":
    main()

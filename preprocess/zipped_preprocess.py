import json
import re
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch

from .frame_extract import pull_video_items
from .border import crop_valid, crop_bounds

# ----------------- Dataset roots -----------------
TRAIN_ROOT = Path(r"D:\수어 영상\수어 영상\1.Training")
VAL_ROOT   = Path(r"D:\수어 영상\수어 영상\2.Validation")

_STEM_RE = re.compile(r"^NIA_SL_WORD(?P<w>\d{4})_REAL(?P<p>\d{2})_(?P<a>[DFLRU])$")


# ----------------- Video / crop constants -----------------
IMG_W = 1920            # legacy
IMG_H = 1080            # legacy
MARGIN_PX = 40
STEP = 1
OUT_SIZE = 256          # your cached size target

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # RGB
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # RGB


def parse_stem(stem: str) -> tuple[int, int, str]:
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Bad stem format: {stem}")
    return int(m.group("w")), int(m.group("p")), m.group("a")


def roots_for_signer(p: int) -> tuple[Path, Path, Path]:
    """
    Returns (VIDEO_ROOT, MORPHEME_ROOT, KEYPOINT_ROOT_BASE) for signer p.

    Train (1..16):
      video:    1.Training/[원천]{q:02d}_real_word_video/{p:02d}-1    where q=2p
      morpheme: 1.Training/[라벨]01_real_word_morpheme/morpheme/{p:02d}
      keypoint: 1.Training/[라벨]{p:02d}_real_word_keypoint/{p:02d}

    Val (17..18):
      video:    2.Validation/[원천]01_real_word_video/WORD/{p:02d}-1
      morpheme: 2.Validation/[라벨]01_real_word_morpheme/morpheme/{p:02d}
      keypoint: 2.Validation/[라벨]09_real_word_keypoint/keypoint/{p:02d}
    """
    if p in (17, 18):
        video_root    = VAL_ROOT / fr"[원천]01_real_word_video\WORD\{p:02d}-1"
        morpheme_root = VAL_ROOT / fr"[라벨]01_real_word_morpheme\morpheme\{p:02d}"
        keypoint_root = VAL_ROOT / fr"[라벨]09_real_word_keypoint\keypoint\{p:02d}"
        return video_root, morpheme_root, keypoint_root

    q = 2 * p
    video_root    = TRAIN_ROOT / fr"[원천]{q:02d}_real_word_video\{p:02d}-1"
    morpheme_root = TRAIN_ROOT / fr"[라벨]01_real_word_morpheme\morpheme\{p:02d}"
    keypoint_root = TRAIN_ROOT / fr"[라벨]{p:02d}_real_word_keypoint\{p:02d}"
    return video_root, morpheme_root, keypoint_root


def load_morpheme(morpheme_path: Path) -> Tuple[float, float, str]:
    """
    Extract start/end in seconds + label from *_morpheme.json.
    """
    d = json.loads(morpheme_path.read_text(encoding="utf-8"))
    item = d["data"][0]
    start_s = float(item["start"])
    end_s = float(item["end"])
    label = str(item["attributes"][0]["name"])
    return start_s, end_s, label


def frame_to_tensor(
    frame_bgr: np.ndarray,
    x1: int,
    x2: int,
    normalise_imagenet: bool = False,
    img_h: int = 1080
) -> torch.Tensor:
    """
    Crop full height using [x1:x2], resize to OUT_SIZE x OUT_SIZE.
    Returns FloatTensor CHW in RGB range [0,1] or ImageNet normalised.
    """
    crop = frame_bgr[:, x1:x2]  # (H, H, 3) BGR uint8 if square crop
    if crop.shape[1] != img_h or crop.shape[0] != img_h:
        raise ValueError(f"Unexpected crop shape: {crop.shape}, bounds=({x1},{x2})")

    crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    crop_rgb = crop[:, :, ::-1].astype(np.float32) / 255.0

    if normalise_imagenet:
        crop_rgb = (crop_rgb - IMAGENET_MEAN) / IMAGENET_STD

    chw = np.transpose(crop_rgb, (2, 0, 1))
    return torch.from_numpy(chw)


def _zip_candidates_from_keypoint_root(keypoint_root: Path) -> List[Path]:
    """
    Try a handful of plausible zip locations/names derived from a keypoint root directory.
    This is intentionally permissive.

    Examples it will catch:
      ...\keypoint\17.zip
      ...\keypoint.zip
      ...\[라벨]09_real_word_keypoint.zip
      ...\[라벨]09_real_word_keypoint\keypoint.zip
    """
    cands: List[Path] = []

    # direct "root.zip"
    cands.append(keypoint_root.with_suffix(".zip"))

    # parent "name.zip"
    cands.append(keypoint_root.parent / f"{keypoint_root.name}.zip")

    # parent folder zipped
    cands.append(keypoint_root.parent.with_suffix(".zip"))

    # grandparent folder zipped (e.g., "[라벨]09_real_word_keypoint.zip")
    cands.append(keypoint_root.parent.parent.with_suffix(".zip"))

    # keypoint_root/..../keypoint.zip
    cands.append(keypoint_root.parent / "keypoint.zip")

    # de-duplicate while preserving order
    seen = set()
    out = []
    for p in cands:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _extract_keypoints_for_stem_from_zip(
    zip_path: Path,
    stem: str,
    out_stem_dir: Path
) -> int:
    """
    Extract only keypoint json files for `stem` into `out_stem_dir`.
    Returns number of files extracted.

    Strategy:
      - scan zip members, keep those whose basename matches:
          f"{stem}_{frame_idx:012d}_keypoints.json"
      - write them under out_stem_dir with their basename

    This avoids needing to know the zip's internal directory structure.
    """
    out_stem_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if base.startswith(stem + "_") and base.endswith("_keypoints.json"):
                # Extract by reading and writing to our desired path
                target = out_stem_dir / base
                with z.open(name, "r") as fsrc:
                    data = fsrc.read()
                target.write_bytes(data)
                extracted += 1

    return extracted


def _resolve_keypoint_dir_or_extract(
    keypoint_root: Path,
    stem: str
) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """
    Returns (keypoint_dir_for_stem, tempdir_handle_or_None).

    - If keypoint_root/stem exists: return it (no tempdir).
    - Else: search for a plausible zip, extract only this stem's jsons into a tempdir,
            and return that tempdir/stem directory.

    The returned TemporaryDirectory (if any) must be kept alive until preprocessing finishes.
    """
    keypoint_dir = keypoint_root / stem
    if keypoint_dir.exists():
        return keypoint_dir, None

    # Fallback: look for a zip
    for zpath in _zip_candidates_from_keypoint_root(keypoint_root):
        if not zpath.exists():
            continue

        tmp = tempfile.TemporaryDirectory(prefix=f"kp_{stem}_")
        out_stem_dir = Path(tmp.name) / stem
        n = _extract_keypoints_for_stem_from_zip(zpath, stem, out_stem_dir)
        if n > 0:
            return out_stem_dir, tmp

        # no files for this stem in that zip -> discard temp and keep searching
        tmp.cleanup()

    raise FileNotFoundError(
        f"Missing keypoint folder and no usable zip found for stem {stem}. "
        f"Expected dir: {keypoint_dir} or a zip near: {keypoint_root}"
    )


def preprocess_stem(
    stem: str,
    step: int = 1,
    margin_px: int = 40,
    normalise_imagenet: bool = False
) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    """
    End-to-end for a single stem.

    Returns:
      clip: (T,3,OUT_SIZE,OUT_SIZE) float32
      kept_indices: original frame indices kept
      meta: label + skip stats
    """
    _, p, _ = parse_stem(stem)
    VIDEO_ROOT, MORPHEME_ROOT, KEYPOINT_ROOT = roots_for_signer(p)

    video_path = VIDEO_ROOT / f"{stem}.mp4"
    morpheme_path = MORPHEME_ROOT / f"{stem}_morpheme.json"

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not morpheme_path.exists():
        raise FileNotFoundError(f"Missing morpheme: {morpheme_path}")

    # Resolve keypoints: directory or zip-extracted temp directory
    keypoint_dir, kp_tmp = _resolve_keypoint_dir_or_extract(KEYPOINT_ROOT, stem)

    start_s, end_s, label = load_morpheme(morpheme_path)
    missing_kp = 0
    discarded_bounds = 0

    img_w = None
    img_h = None

    def transform(frame_bgr: np.ndarray, frame_idx: int) -> Optional[torch.Tensor]:
        nonlocal missing_kp, discarded_bounds, img_w, img_h

        if img_h is None or img_w is None:
            img_h, img_w = frame_bgr.shape[:2]
            if img_w < img_h:
                raise ValueError(
                    f"Expected img_w >= img_h for horizontal square crop, got {img_w}x{img_h}"
                )

        kp_path = keypoint_dir / f"{stem}_{frame_idx:012d}_keypoints.json"
        if not kp_path.exists():
            missing_kp += 1
            return None

        ok, direction, error = crop_valid(kp_path, img_w, img_h, margin_px=margin_px)
        bounds = crop_bounds(ok, direction, error, img_w=img_w, img_h=img_h, margin_px=margin_px)
        if bounds is None:
            discarded_bounds += 1
            return None

        x1, x2 = bounds
        return frame_to_tensor(
            frame_bgr,
            x1,
            x2,
            img_h=img_h,
            normalise_imagenet=normalise_imagenet,
        )

    try:
        tensors, kept_indices, fps, requested_indices, decode_fail = pull_video_items(
            video_path,
            start_s,
            end_s,
            step=step,
            transform=transform,
            seek_to_start=True,
        )

        if not tensors:
            raise RuntimeError(
                f"No usable frames for {stem}. requested={len(requested_indices)} "
                f"missing_kp={missing_kp} discarded={discarded_bounds}"
            )

        clip = torch.stack(tensors, dim=0)

        meta = {
            "stem": stem,
            "label": label,
            "start_s": start_s,
            "end_s": end_s,
            "fps": fps,
            "requested_frames": len(requested_indices),
            "kept_frames": len(kept_indices),
            "missing_keypoints_frames": missing_kp,
            "discarded_size_or_oob_frames": discarded_bounds,
            "decode_fail": decode_fail,
            "keypoints_source": "dir" if kp_tmp is None else "zip",
            "keypoints_dir": str(keypoint_dir),
        }
        return clip, kept_indices, meta

    finally:
        # Ensure tempdir cleanup if we had to extract from zip
        if kp_tmp is not None:
            kp_tmp.cleanup()


def batch_check(start_word: int, n_words: int, step: int, margin_px: int) -> None:
    angles = ["D", "F", "L", "R", "U"]
    # NOTE: this helper still uses REAL01; adjust if you want multi-signer checks.
    for w in range(start_word, start_word + n_words):
        for a in angles:
            stem = f"NIA_SL_WORD{w:04d}_REAL01_{a}"
            try:
                clip, index, meta = preprocess_stem(stem, step=step, margin_px=margin_px, normalise_imagenet=True)
                print(
                    f"{stem} label={meta['label']} "
                    f"kept={meta['kept_frames']}/{meta['requested_frames']} "
                    f"missing_kp={meta['missing_keypoints_frames']} "
                    f"discarded={meta['discarded_size_or_oob_frames']} "
                    f"kp_src={meta['keypoints_source']}"
                )
            except Exception as e:
                print(f"[FAIL] {stem}: {e}")


if __name__ == "__main__":
    batch_check(start_word=1501, n_words=30, step=STEP, margin_px=MARGIN_PX)

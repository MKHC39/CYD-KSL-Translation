import json
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch

from frame_extract import pull_video_items
from border import crop_valid, crop_bounds
import re

from pathlib import Path


_STEM_RE = re.compile(r"^NIA_SL_WORD(?P<w>\d{4})_REAL(?P<p>\d{2})_(?P<a>[DFLRU])$")


# ----------------- Video / crop constants -----------------
IMG_W = 1920            # legacy
IMG_H = 1080            # legacy
MARGIN_PX = 40          # passed as arg
STEP = 1                # passed as arg
# ResNet input size
OUT_SIZE = 256

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # RGB
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # RGB


def parse_stem(stem: str) -> tuple[int, int, str]:
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Bad stem format: {stem}")
    return int(m.group("w")), int(m.group("p")), m.group("a")

def ISLR_roots(p: int, val_root: Path, train_root: Path) -> tuple[Path, Path, Path]:
    """
    Returns (VIDEO_ROOT, MORPHEME_ROOT, KEYPOINT_ROOT) for signer p.

    Train signers: 1..16 in 1.Training
    Val signers:   17..18 in 2.Validation
    """
    if p in (17, 18):
        video_root    = val_root / "[원천]01_real_word_video" / "WORD" / f"{p:02d}-1"
        morpheme_root = val_root / "[라벨]01_real_word_morpheme" / "morpheme" / f"{p:02d}"
        keypoint_root = val_root / "[라벨]09_real_word_keypoint" / "keypoint" / f"{p:02d}"
        return video_root, morpheme_root, keypoint_root

    q = 2 * p
    video_root    = train_root / f"[원천]{q:02d}_real_word_video" / f"{p:02d}-1"
    morpheme_root = train_root / "[라벨]01_real_word_morpheme" / "morpheme"/ f"{p:02d}"
    keypoint_root = train_root / f"[라벨]{p:02d}_real_word_keypoint" / f"{p:02d}"
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
        img_h:int = 1080
    ) -> torch.Tensor:
    """
    Crop full height using [x1:x2], resize to 224x224.
    Returns FloatTensor CHW in RGB range [0,1] or normalised

    :param frame_bgr: BGR frame
    :param x1: int
    :param x2: int
    :param normalise_imagenet: bool
    :param img_h: int
    :return: Tensor
    """
    # full height crop
    crop = frame_bgr[:, x1:x2]  # (1080,1080,3) BGR uint8
    if crop.shape[1] != img_h or crop.shape[0] != img_h:
        raise ValueError(f"Unexpected crop shape: {crop.shape}, bounds=({x1},{x2})")

    # downscale to 224x224
    crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

    # BGR -> RGB
    crop_rgb = crop[:, :, ::-1].astype(np.float32) / 255.0

    if normalise_imagenet:
        crop_rgb = (crop_rgb - IMAGENET_MEAN) / IMAGENET_STD


    # HWC -> CHW
    chw = np.transpose(crop_rgb, (2, 0, 1))
    return torch.from_numpy(chw)  # (3,224,224) float32


def preprocess_stem(
        stem: str,
        data_root: Path,
        step: int = 1,
        margin_px: int = 40,
        normalise_imagenet: bool = False,
    ) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    """
    End-to-end for a single stem:
      stem: "NIA_SL_WORD1501_REAL01_D"

    Returns:
      clip: (T,3,224,224) float32
      kept_indices: original frame indices kept
      meta: label + skip stats
    """

    _, p, _ = parse_stem(stem)
    train_root = data_root / "1.Training"
    val_root = data_root / "2.Validation"
    video_root, morpheme_root, keypoint_root = ISLR_roots(p, train_root=train_root, val_root=val_root)

    video_path = video_root / f"{stem}.mp4"
    morpheme_path = morpheme_root / f"{stem}_morpheme.json"
    keypoint_dir = keypoint_root / stem

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not morpheme_path.exists():
        raise FileNotFoundError(f"Missing morpheme: {morpheme_path}")
    """
    if not keypoint_dir.exists():
        raise FileNotFoundError(f"Missing keypoint folder: {keypoint_dir}")
    """

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
                # reject this video early; you can also "return None" but that hides the error
                raise ValueError(f"Expected img_w >= img_h for horizontal square crop, got {img_w}x{img_h}")

        kp_path = keypoint_dir / f"{stem}_{frame_idx:012d}_keypoints.json"
        """
        if not kp_path.exists():
            missing_kp += 1
            return None             Skip kp detection
        """

        # ok, direction, error = crop_valid(kp_path, img_w, img_h, margin_px=margin_px)
        ok, direction, error = (True, 'ok', 0)
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

    clip = torch.stack(tensors, dim=0)  # (T,3,224,224)

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
    }
    return clip, kept_indices, meta


def batch_check(start_word: int, n_words: int, step: int, margin_px: int, img_w: int, img_h: int) -> None:
    angles = ["D", "F", "L", "R", "U"]
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
                    f"decode_fail={meta['decode_fail']}"
                )
            except Exception as e:
                print(f"[FAIL] {stem}: {e}")


if __name__ == "__main__":
    batch_check(start_word=1501, n_words=30, step=STEP, margin_px=MARGIN_PX, img_w= IMG_W, img_h=IMG_H)

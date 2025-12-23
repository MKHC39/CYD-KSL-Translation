import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch

from .frame_extract import pull_video_frames
from .border import crop_valid, crop_bounds

# ----------------- Dataset roots  -----------------
TRAIN_ROOT = Path(r"C:\Users\CHOI\Downloads\KSL Word DataSet\수어 영상\1.Training")
VIDEO_ROOT = TRAIN_ROOT / r"[원천]01_real_word_video\01"
MORPHEME_ROOT = TRAIN_ROOT / r"[라벨]01_real_word_morpheme\morpheme\01"
KEYPOINT_ROOT = TRAIN_ROOT / r"[라벨]01_real_word_keypoint\01"

# ----------------- Video / crop constants -----------------
IMG_W = 1920            # legacy
IMG_H = 1080            # legacy
MARGIN_PX = 40
STEP = 1
# ResNet input size
OUT_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # RGB
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # RGB


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
        normalise_imagenet: bool = True,
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
    # full height crop, fixed width 1080
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
        step: int = 1,
        margin_px: int = 40,
        normalise_imagenet: bool = True
    ) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    """
    End-to-end for a single stem:
      stem: "NIA_SL_WORD1501_REAL01_D"

    Returns:
      clip: (T,3,224,224) float32
      kept_indices: original frame indices kept
      meta: label + skip stats
    """

    video_path = VIDEO_ROOT / f"{stem}.mp4"
    morpheme_path = MORPHEME_ROOT / f"{stem}_morpheme.json"
    keypoint_dir = KEYPOINT_ROOT / stem

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not morpheme_path.exists():
        raise FileNotFoundError(f"Missing morpheme: {morpheme_path}")
    if not keypoint_dir.exists():
        raise FileNotFoundError(f"Missing keypoint folder: {keypoint_dir}")

    start_s, end_s, label = load_morpheme(morpheme_path)
    frames, indices, fps = pull_video_frames(video_path, start_s, end_s, step=step)

    if not frames:
        raise ValueError(f"No frames extracted for {stem}")

    img_h, img_w = frames[0].shape[:2]

    if img_w < img_h:
        raise ValueError(f"Expected img_w >= img_h for horizontal square crop, got {img_w}x{img_h}")

    missing_kp = 0
    discarded_bounds = 0

    tensors: List[torch.Tensor] = []
    kept_indices: List[int] = []

    for frame_bgr, frame_idx in zip(frames, indices):
        kp_path = keypoint_dir / f"{stem}_{frame_idx:012d}_keypoints.json"
        if not kp_path.exists():
            missing_kp += 1
            continue

        ok, direction, error = crop_valid(kp_path, img_w, img_h, margin_px=margin_px)
        bounds = crop_bounds(ok, direction, error, img_w=img_w, img_h=img_h, margin_px=margin_px)

        if bounds is None:
            discarded_bounds += 1
            continue

        x1, x2 = bounds
        tensors.append(frame_to_tensor(frame_bgr, x1, x2, img_h=img_h, normalise_imagenet=normalise_imagenet))
        kept_indices.append(frame_idx)

    if not tensors:
        raise RuntimeError(
            f"No usable frames for {stem}. requested={len(indices)} missing_kp={missing_kp} discarded={discarded_bounds}"
        )

    clip = torch.stack(tensors, dim=0)  # (T,3,224,224)

    meta = {
        "stem": stem,
        "label": label,
        "start_s": start_s,
        "end_s": end_s,
        "fps": fps,
        "requested_frames": len(indices),
        "kept_frames": len(kept_indices),
        "missing_keypoints_frames": missing_kp,
        "discarded_size_or_oob_frames": discarded_bounds,
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
                    f"discarded={meta['discarded_size_or_oob_frames']}"
                )
            except Exception as e:
                print(f"[FAIL] {stem}: {e}")


if __name__ == "__main__":
    batch_check(start_word=1501, n_words=50, step=STEP, margin_px=MARGIN_PX, img_w= IMG_W, img_h=IMG_H)

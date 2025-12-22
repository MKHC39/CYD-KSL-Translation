from pathlib import Path
from typing import Iterable

import numpy as np
import cv2
import torch

from preprocess_clip import preprocess_stem, IMAGENET_MEAN, IMAGENET_STD


# ---------- EDITABLE OUTPUT ROOT ----------
PREVIEW_ROOT = Path(r"C:\Users\CHOI\Downloads\KSL Word DataSet\수어 영상\1.Training\preview_test")


def write_jpg_unicode_safe(path: Path, bgr_uint8: np.ndarray, quality: int = 95) -> None:
    ok, buf = cv2.imencode(".jpg", bgr_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for: {path}")
    path.write_bytes(buf.tobytes())


def denormalise_imagenet(clip: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    x = clip.detach().cpu().to(torch.float32)
    x = x * std + mean
    return torch.clamp(x, 0.0, 1.0)


def save_clip_as_jpgs(stem: str, clip_rgb01: torch.Tensor, kept_indices: list[int]) -> int:
    out_dir = PREVIEW_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, frame_idx in enumerate(kept_indices):
        img = clip_rgb01[i].permute(1, 2, 0).numpy()  # HWC RGB float [0,1]
        rgb_uint8 = (img * 255.0 + 0.5).astype(np.uint8)
        bgr_uint8 = rgb_uint8[:, :, ::-1]
        out_path = out_dir / f"{frame_idx:012d}.jpg"
        write_jpg_unicode_safe(out_path, bgr_uint8)

    return len(kept_indices)


def iter_stems(start_word: int, n_words: int, angles: Iterable[str]) -> Iterable[str]:
    for w in range(start_word, start_word + n_words):
        for a in angles:
            yield f"NIA_SL_WORD{w:04d}_REAL01_{a}"


def main(start_word: int, n_words: int, angles: Iterable[str], step: int, shift_margin_px: int, save_previews: bool, img_w: int, img_h: int) -> None:
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)

    for stem in iter_stems(start_word, n_words, angles):
        try:
            clip, kept_indices, meta = preprocess_stem(
                stem,
                step=step,
                margin_px=shift_margin_px,
                normalise_imagenet=True,
                img_w=img_w,
                img_h=img_h,
            )

            saved = 0
            if save_previews:
                clip_rgb01 = denormalise_imagenet(clip)
                saved = save_clip_as_jpgs(stem, clip_rgb01, kept_indices)

            # --- one-line summary (similar to batch_check) ---
            print(
                f"{stem} label={meta['label']} "
                f"kept={meta['kept_frames']}/{meta['requested_frames']} "
                f"missing_kp={meta['missing_keypoints_frames']} "
                f"discarded={meta['discarded_size_or_oob_frames']} "
                f"saved_jpg={saved}"
            )

        except Exception as e:
            print(f"[FAIL] {stem}: {e}")


if __name__ == "__main__":
    # ------------- EDIT THESE VARIABLES -------------
    START_WORD = 1501
    N_WORDS = 10
    ANGLES = ["D", "F", "L", "R", "U"]

    STEP = 5
    SHIFT_MARGIN_PX = 40  # this is your crop_bounds shift margin
    SAVE_PREVIEWS = True
    IMG_W, IMG_H = 1920, 1080
    # -----------------------------------------------

    main(
        start_word=START_WORD,
        n_words=N_WORDS,
        angles=ANGLES,
        step=STEP,
        shift_margin_px=SHIFT_MARGIN_PX,
        save_previews=SAVE_PREVIEWS,
        img_w=IMG_W,
        img_h=IMG_H,
    )

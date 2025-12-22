import json
from pathlib import Path
from typing import Tuple, List

from frame_extract import pull_video_frames
from border import box_outline


# ---- ROOTS (from your message) ----
TRAIN_ROOT = Path(r"C:\Users\CHOI\Downloads\KSL Word DataSet\수어 영상\1.Training")

VIDEO_ROOT = TRAIN_ROOT / r"[원천]01_real_word_video\01"
MORPHEME_ROOT = TRAIN_ROOT / r"[라벨]01_real_word_morpheme\morpheme\01"
KEYPOINT_ROOT = TRAIN_ROOT / r"[라벨]01_real_word_keypoint\01"

IMG_W, IMG_H = 1920, 1080
STEP = 5


def morpheme_data(morpheme_path: Path) -> Tuple[float, float, str]:
    """
    Matches your sample schema:
      {"data": [{"start": 1.743, "end": 3.103, "attributes":[{"name":"..."}]}]}
    """
    d = json.loads(morpheme_path.read_text(encoding="utf-8"))
    item = d["data"][0]
    start_s = float(item["start"])
    end_s = float(item["end"])
    label = str(item["attributes"][0]["name"])
    return start_s, end_s, label


def main():
    # ---- EDIT THIS ONLY ----
    stem = "NIA_SL_WORD1501_REAL01_D"  # without extension
    # ------------------------

    video_path = VIDEO_ROOT / f"{stem}.mp4"
    morpheme_path = MORPHEME_ROOT / f"{stem}_morpheme.json"
    keypoint_dir = KEYPOINT_ROOT / stem

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not morpheme_path.exists():
        raise FileNotFoundError(f"Missing morpheme json: {morpheme_path}")
    if not keypoint_dir.exists():
        raise FileNotFoundError(f"Missing keypoint folder: {keypoint_dir}")

    start_s, end_s, label = morpheme_data(morpheme_path)

    frames, indices, fps = pull_video_frames(video_path, start_s, end_s, step=STEP)

    print(f"stem: {stem}")
    print(f"label: {label}")
    print(f"start_s={start_s}, end_s={end_s}, step={STEP}, fps={fps}")
    print(f"extracted_frames={len(frames)}")
    print(f"indices (first 10): {indices[:10]}")
    print(f"indices (last 10):  {indices[-10:]}")

    if frames:
        print(f"first_frame_shape={frames[0].shape} dtype={frames[0].dtype}")

    # bbox checks for the extracted indices
    big_area = 0
    max_area = -1
    max_area_info = None

    # show only a few per-frame prints so output stays readable
    PREVIEW = 5
    preview_printed = 0

    for frame_idx in indices:
        kp_path = keypoint_dir / f"{stem}_{frame_idx:012d}_keypoints.json"
        if not kp_path.exists():
            raise FileNotFoundError(f"Missing keypoints json for frame {frame_idx}: {kp_path}")

        x1, y1, x2, y2 = box_outline(kp_path, IMG_W, IMG_H, margin_px=20)
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        if area > 1_000_000:
            big_area += 1

        if area > max_area:
            max_area = area
            max_area_info = (frame_idx, (x1, y1, x2, y2), area)

        if preview_printed < PREVIEW:
            print(f"frame={frame_idx:012d} bbox={(x1,y1,x2,y2)} area={area}")
            preview_printed += 1

    print(f"large_area_frames(>1,000,000): {big_area}/{len(indices)}")
    if max_area_info is not None:
        fi, bb, ar = max_area_info
        print(f"max_area: frame={fi:012d} bbox={bb} area={ar}")


if __name__ == "__main__":
    main()

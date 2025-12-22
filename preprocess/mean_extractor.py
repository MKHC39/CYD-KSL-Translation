import json
from pathlib import Path
from typing import Tuple

import numpy as np


def mean_value(
    keypoints_json: Path
) -> list:
    """
    One-pass bbox extraction:
      - reads x,y from each triple
      - updates min/max immediately
      - no point list allocated
    Rule A: keep all finite points; ignore confidence; drop pose.
    Returns inclusive (x1,y1,x2,y2) clamped to frame.
    """
    d = json.loads(keypoints_json.read_text(encoding="utf-8"))
    people = d["people"]

    mean = []

    for keypoints in ("face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
        arr = people[keypoints]
        n = len(arr)
        if n % 3 != 0:
            raise ValueError(f"{keypoints} length not divisible by 3: {n} in {keypoints_json.name}")

        x_array = []
        y_array = []
        for i in range(0, n, 3):
            x = arr[i]
            y = arr[i + 1]
            # ignore arr[i+2] (confidence)

            x_array.append(x)
            y_array.append(y)

        mean_x = np.mean(x_array)
        mean_y = np.mean(y_array)

        mean.append(Tuple[str(keypoints), mean_x, mean_y])


    return mean

def main():
    stem = "NIA_SL_WORD1501_REAL01_D"
    TRAIN_ROOT = Path(r"C:\Users\CHOI\Downloads\KSL Word DataSet\수어 영상\1.Training")

    VIDEO_ROOT = TRAIN_ROOT / r"[원천]01_real_word_video\01"
    MORPHEME_ROOT = TRAIN_ROOT / r"[라벨]01_real_word_morpheme\morpheme\01"
    KEYPOINT_ROOT = TRAIN_ROOT / r"[라벨]01_real_word_keypoint\01"

    keypoint_dir = KEYPOINT_ROOT / stem

    if not keypoint_dir.exists():
        raise FileNotFoundError(f"Missing keypoint folder: {keypoint_dir}")

    for frame_idx in range(0, 100):
        kp_path = keypoint_dir / f"{stem}_{frame_idx:012d}_keypoints.json"
        if not kp_path.exists():
            raise FileNotFoundError(f"Missing keypoints json for frame {frame_idx}: {kp_path}")

        print(mean_value(kp_path))

if __name__ == "__main__":
    main()
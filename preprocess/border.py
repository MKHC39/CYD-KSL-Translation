import json
import math
from pathlib import Path
from typing import Tuple


def box_outline(
    keypoints_json: Path,
    img_w: int,
    img_h: int,
    margin_px: int = 20,
) -> Tuple[int, int, int, int]:
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

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for keypoints in ("face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
        arr = people[keypoints]
        n = len(arr)
        if n % 3 != 0:
            raise ValueError(f"{keypoints} length not divisible by 3: {n} in {keypoints_json.name}")

        for i in range(0, n, 3):
            x = arr[i]
            y = arr[i + 1]
            # ignore arr[i+2] (confidence)

            # compare min max
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y):
                if x < min_x: min_x = x
                if y < min_y: min_y = y
                if x > max_x: max_x = x
                if y > max_y: max_y = y

    if min_x == float("inf"):
        raise ValueError(f"No finite keypoints found in {keypoints_json}")

    # inclusive integer bbox
    x1 = int(math.floor(min_x)) - margin_px
    y1 = int(math.floor(min_y)) - margin_px
    x2 = int(math.ceil(max_x)) + margin_px
    y2 = int(math.ceil(max_y)) + margin_px

    # clamp inclusive
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid bbox after clamping: {(x1, y1, x2, y2)}")

    return x1, y1, x2, y2

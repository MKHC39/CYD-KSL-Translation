import json
import math
from pathlib import Path
from typing import Tuple, Optional, Literal


def crop_valid(
    keypoints_json: Path,
    img_w: int,
    img_h: int,
    margin_px: int = 40,
) -> Tuple[bool, str, int]:
    """
    Checks if square crop exceeds keypoint value

    :param keypoints_json:
    :param img_w:
    :param img_h:
    :param margin_px:

    :return: (ok, crop_direction, error_factor)
    """
    d = json.loads(keypoints_json.read_text(encoding="utf-8"))
    people = d["people"]

    min_x = float("inf")
    max_x = float("-inf")

    crop_factor = int((img_w - img_h)/2)
    max_crop = img_w - crop_factor              # 1500
    min_crop = crop_factor                      # 420


    for keypoints in ("face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
        arr = people[keypoints]
        n = len(arr)
        if n % 3 != 0:
            raise ValueError(f"{keypoints} length not divisible by 3: {n} in {keypoints_json.name}")

        for i in range(0, n, 3):
            x = arr[i]

            # compare min max
            if isinstance(x, (int, float)) and math.isfinite(x):
                if x < min_x: min_x = x
                if x > max_x: max_x = x

    if min_x == float("inf"):
        raise ValueError(f"No finite keypoints found in {keypoints_json}")

    # inclusive integer bbox
    x1 = int(math.floor(min_x))
    x2 = int(math.ceil(max_x))

    # clamp inclusive
    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))

    if x2 < x1:
        raise ValueError(f"Invalid x after clamping: {(x1, x2)}")

    # check if squaring crops keypoint
    if x1 < min_crop or x2 >= max_crop:
        crop_check = False

        # which direction fails?
        if x2-x1+1+margin_px >= img_h:
            crop_direction = "size"
            error_factor = 0
        elif x1 < min_crop:
            crop_direction = "min"
            error_factor = min_crop - x1
        elif x2 >= max_crop:
            crop_direction = "max"
            error_factor = x2+1 - max_crop
        else:
            raise ValueError(f"x-coord outside crop: {(x1, x2)}")

    else:
        crop_check = True
        crop_direction = "ok"
        error_factor = 0

    return crop_check, crop_direction, error_factor


def crop_bounds(
        ok: bool,
        direction: Literal["min", "max", "size", "ok"],
        error_factor: int,
        img_w: int = 1920,
        img_h: int = 1080,
        margin_px: int = 40,
) -> Optional[Tuple[int, int]]:

    """
    Computes the x coordinate of the crop boundary, shifting if necessary.

    :param ok:
    :param direction: {"min", "max", "size", "ok"}
    :param error_factor:
    :param img_w:
    :param img_h:
    :param margin_px:

    :return: (crop_x1, crop_x2)

    Returns the x coord of the crop boundary.
    """

    crop_w = img_h                              # 1080
    crop_factor = int((img_w - img_h) / 2)
    max_crop = img_w - crop_factor              # 1500
    min_crop = crop_factor                      # 420

    if direction == "size":
        return None

    if ok or direction == "ok":
        return min_crop, max_crop

    if direction not in ("min", "max"):
        raise ValueError(f"Unexpected direction: {direction}")

    shift = error_factor + margin_px       # shift crop direction
    if direction == "min":
        new_min = min_crop - shift
    else:  # "max"
        new_min = min_crop + shift

    new_max = new_min + crop_w

    # If shifting would push outside video bounds, set to within bounds
    if new_min < 0:
        new_min = 0
        new_max = new_min + crop_w
    elif new_max > img_w:
        new_max = img_w
        new_min = new_max - crop_w

    return new_min, new_max



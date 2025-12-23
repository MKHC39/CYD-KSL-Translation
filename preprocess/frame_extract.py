import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def time_to_frame(start_s: float, end_s: float, fps: float) -> Tuple[int, int]:
    """
    Inclusive [start_s, end_s] in time => frames i such that start_s <= i/fps <= end_s

    return: (starting frame, ending frame)
    """
    start_f = int(math.ceil(start_s * fps))
    end_f = int(math.floor(end_s * fps))
    return start_f, end_f


def frame_index(start_f: int, end_f: int, step: int = 1) -> List[int]:
    """
    - indices = start_f, start_f+step, start_f+2*step, ... <= end_f
    - if the last index is not end_f, append end_f
    """
    if end_f < start_f:
        return []

    if step <= 0:
        raise ValueError(f"Invalid step reported: {step}")    # ??

    index = list(range(start_f, end_f + 1, step))


    if index[-1] != end_f:
        index.append(end_f)

    return index


def frame_extractor(
    video_path: Path,
    xtract_index: List[int],
) -> List[np.ndarray]:
    """
    Sequential decode; keeps frames whose 0-based index is in xtract_ind.
    Frames returned as BGR uint8 (OpenCV default).
    """
    if not xtract_index:
        return []

    keep_set = set(xtract_index)
    max_idx = max(xtract_index)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames: List[np.ndarray] = []
    curr_frame = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if curr_frame in keep_set:
            frames.append(frame)

        if curr_frame >= max_idx:
            break

        curr_frame += 1

    cap.release()

    return frames


def pull_video_frames(
    video_path: Path,
    start_s: float,
    end_s: float,
    step: int = 1,
) -> Tuple[List[np.ndarray], List[int], float]:
    """
      - compute start_f/end_f from seconds (inclusive)
      - build indices using step-5-from-start + include end_f
      - sequentially decode and extract those frames

      :param video_path:
      :param start_s:
      :param end_s:
      :param step:
      :return: (frames, indices, fps)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS reported: {fps}")

    start_f, end_f = time_to_frame(start_s, end_s, fps)
    if end_f < start_f:
        raise ValueError(f"Bad time window: start_f={start_f}, end_f={end_f}")

    indices = frame_index(start_f, end_f, step=step)
    frames = frame_extractor(video_path, indices)
    return frames, indices, fps

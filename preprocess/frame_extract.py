import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def time_to_frame(start_s: float, end_s: float, fps: float) -> Tuple[int, int]:
    """
    Inclusive [start_s, end_s] in time => frames i such that start_s <= i/fps <= end_s
    => start_f = ceil(start_s*fps), end_f = floor(end_s*fps)
    """
    start_f = int(math.ceil(start_s * fps))
    end_f = int(math.floor(end_s * fps))
    return start_f, end_f


def frame_index(start_f: int, end_f: int, step: int = 5) -> List[int]:
    """
    - indices = start_f, start_f+step, start_f+2*step, ... <= end_f
    - if the last index is not end_f, append end_f
    """
    if end_f < start_f:
        return []

    index = list(range(start_f, end_f + 1, step))
    if len(index)==0:
        return [end_f]  # ??

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
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index in keep_set:
            frames.append(frame)

        if frame_index >= max_idx:
            break

        frame_index += 1

    cap.release()

    # Sanity: ensure we actually got everything we asked for (helps catch decode/length issues)
    # This check is Inferred (engineering safety), not guaranteed necessary.
    if len(frames) != len(xtract_index):
        missing = [i for i in xtract_index if i not in keep_set]  # should be empty logically
        # More useful: detect which requested indices weren't captured
        # We can reconstruct captured indices by re-decoding with a counter if needed.
        raise RuntimeError(
            f"Expected {len(xtract_index)} frames, got {len(frames)}. "
            f"Video may be shorter than requested max_idx={max_idx}."
        )

    return frames


def pull_video_frames(
    video_path: Path,
    start_s: float,
    end_s: float,
    step: int = 5,
) -> Tuple[List[np.ndarray], List[int], float]:
    """
    Full pipeline:
      - compute start_f/end_f from seconds (inclusive)
      - build indices using step-5-from-start + include end_f
      - sequentially decode and extract those frames
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

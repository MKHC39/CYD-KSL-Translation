import math
from pathlib import Path
from typing import List, Tuple, Callable, Optional, TypeVar

import cv2
import numpy as np

T = TypeVar("T")

def time_to_frame(start_s: float, end_s: float, fps: float) -> Tuple[int, int]:
    """
    Inclusive [start_s, end_s] in time => frames

    return: (starting frame, ending frame)
    """
    start_f = int(math.floor(start_s * fps))
    end_f = int(math.ceil(end_s * fps))
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


def pull_video_items(
    video_path: Path,
    start_s: float,
    end_s: float,
    step: int,
    transform: Callable[[np.ndarray, int], Optional[T]],
    *,
    seek_to_start: bool = True,
) -> Tuple[List[T], List[int], float, List[int], int]:
    """
    Single-pass video open:
      - opens cv2.VideoCapture ONCE
      - reads fps
      - computes indices
      - sequentially decodes, and for requested frames calls transform(frame_bgr, frame_idx)
      - if transform returns None => skip that frame

        :param video_path: path to video
        :param start_s: starting second
        :param end_s: ending second
        :param step: number of frames to skip
        :param transform: transform function
        :param seek_to_start: Skip to first frame

        :return: (Tensors, kept_indices, fps, requested_indices, decode_fail )

    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS reported: {fps}")

    start_f, end_f = time_to_frame(start_s, end_s, fps)
    if end_f < start_f:
        cap.release()
        raise ValueError(f"Bad time window: start_f={start_f}, end_f={end_f}")

    requested = frame_index(start_f, end_f, step=step)
    if not requested:
        cap.release()
        return [], [], fps, []

    max_idx = max(requested)

    # Optional speed-up: skip decoding frames before start_f
    if seek_to_start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        curr_frame = start_f
    else:
        curr_frame = 0

    items: List[T] = []
    kept_indices: List[int] = []

    j = 0
    n = len(requested)
    decode_fail = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            decode_fail += 1
            break

        if curr_frame == requested[j]:
            out = transform(frame, curr_frame)
            if out is not None:
                items.append(out)
                kept_indices.append(curr_frame)
            j += 1
            if j >= n:
                break  # processed all requested frames

        if curr_frame >= max_idx:
            break
        curr_frame += 1

    cap.release()
    return items, kept_indices, fps, requested, decode_fail



from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import Dataset

from preprocess.preprocess_clip import preprocess_stem


ANGLES = ("D", "F", "L", "R", "U")


@dataclass(frozen=True)
class Sample:
    w: int
    stem: str


def iter_samples(
    w_start: int = 1501,
    w_end: int = 3000,
    angles: Iterable[str] = ANGLES,
) -> Iterable[Sample]:
    """
    Generates samples in the exact format you specified.
    """
    for w in range(w_start, w_end + 1):
        for a in angles:
            stem = f"NIA_SL_WORD{w:04d}_REAL01_{a}"
            yield Sample(w=w, stem=stem)


def build_w_id_maps(w_start: int = 1501, w_end: int = 3000) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    w_to_id: WORD number -> contiguous class id in [0, C-1]
    id_to_w: inverse mapping for decoding predictions
    """
    w_to_id: Dict[int, int] = {}
    id_to_w: Dict[int, int] = {}
    for cid, w in enumerate(range(w_start, w_end + 1)):
        w_to_id[w] = cid
        id_to_w[cid] = w
    return w_to_id, id_to_w


class KSLStemDataset(Dataset):
    """
    Returns:
      clip:   (T,3,224,224) float32 (ImageNet-normalised if preprocess_stem does that)
      y:      int class id (0..C-1) derived from w
      length: int T
      stem:   str (for debugging)
      w:      int word number (for debugging)
    """

    def __init__(
        self,
        w_start: int = 1501,
        w_end: int = 3000,
        angles: Iterable[str] = ANGLES,
        step: int = 5,
        shift_margin_px: int = 40,
        img_w: int = 1920,
        img_h: int = 1080,
        normalise_imagenet: bool = True,
    ):
        if w_end < w_start:
            raise ValueError("w_end must be >= w_start")

        self.w_start = w_start
        self.w_end = w_end
        self.angles = tuple(angles)

        self.step = step
        self.shift_margin_px = shift_margin_px
        self.img_w = img_w
        self.img_h = img_h
        self.normalise_imagenet = normalise_imagenet

        # deterministic sample list
        self.samples: List[Sample] = list(iter_samples(w_start, w_end, self.angles))

        # label mapping (w -> contiguous id)
        self.w_to_id, self.id_to_w = build_w_id_maps(w_start, w_end)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        clip, kept_indices, meta = preprocess_stem(
            s.stem,
            step=self.step,
            margin_px=self.shift_margin_px,
            normalise_imagenet=self.normalise_imagenet,
        )
        y = self.w_to_id[s.w]
        length = int(clip.shape[0])
        return clip, y, length, s.stem, s.w, meta  # meta included for debugging


def collate_pad_time(batch):
    """
    Pads variable-length clips along time dimension.

    Input batch items:
      (clip, y, length, stem, w, meta)

    Output:
      clips:   (B, T_max, 3, 224, 224)
      ys:      (B,)
      lengths: (B,)
      stems:   tuple[str,...]
      ws:      (B,)
      metas:   tuple[dict,...]
    """
    clips, ys, lengths, stems, ws, metas = zip(*batch)

    B = len(clips)
    T_max = max(lengths)
    C, H, W = clips[0].shape[1:]  # clip is (T,3,224,224)

    out = torch.zeros((B, T_max, C, H, W), dtype=clips[0].dtype)

    for i, clip in enumerate(clips):
        T = clip.shape[0]
        out[i, :T] = clip

    ys_t = torch.tensor(ys, dtype=torch.long)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    ws_t = torch.tensor(ws, dtype=torch.long)

    return out, ys_t, lengths_t, stems, ws_t, metas

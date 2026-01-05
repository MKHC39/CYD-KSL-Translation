"""
KSL_pydata.py

Drop-in CorrNet "feeder" for your KSL clips.

Goal:
- Make a class that CorrNet's main.py can instantiate via --feeder,
  and that exposes `collate_fn` like BaseFeeder, so CorrNet's downstream
  pipeline (temporal padding, model forward, decode) can be reused.

Contract (matches CorrNet BaseFeeder):
- __getitem__ -> (video, label, original_info)
  video: FloatTensor (T, 3, H, W)
  label: LongTensor (L,)  (CTC token sequence; for single-gloss clips, L=1)
  original_info: Any (passed through unchanged)

Normalisation:
- preprocess_stem(..., normalise_imagenet=False) returns RGB in [0,1].
- CorrNet normalises video to [-1,1] (it does video.float()/127.5 - 1 on uint8).
  We therefore map [0,1] -> [-1,1] via: video = video * 2 - 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

# Your preprocessing entrypoint (must be importable in your project)

# CorrNet's collate_fn relies on a module-level `kernel_sizes` list
from . import dataloader_video
from utils import video_augmentation

DEFAULT_ANGLES: Tuple[str, ...] = ("D", "F", "L", "R", "U")
DEFAULT_SIGNERS_TRAIN = tuple(range(1, 17))   # 1..16
DEFAULT_SIGNERS_DEV   = (17, 18)

@dataclass(frozen=True)
class Sample:
    stem: str
    w: int
    p: int
    angle: str

def _iter_samples(
    w_start: int,
    w_end: int,
    signers: Sequence[int],
    angles: Sequence[str],
) -> List[Sample]:
    out: List[Sample] = []
    for p in signers:
        for w in range(w_start, w_end + 1):
            for a in angles:
                stem = f"NIA_SL_WORD{w:04d}_REAL{p:02d}_{a}"
                out.append(Sample(stem=stem, w=w, p=p, angle=a))
    return out



class KSLFeeder(Dataset):
    """
    CorrNet-compatible feeder.

    CorrNet main.py calls:
        feeder(gloss_dict=..., kernel_size=..., dataset=..., prefix=..., mode=..., transform_mode=..., **feeder_args)

    This class accepts those args (and ignores what it doesn't need),
    and exposes a class-level `collate_fn` so CorrNet can call:
        DataLoader(..., collate_fn=self.feeder.collate_fn)

    Your KSL-specific controls go in feeder_args:
      - w_start, w_end
      - angles
      - step
      - margin_px
      - split: "none" | "by_word_mod"
      - split_mod_k (int), split_mod_train (set of ints), split_mod_dev (set of ints), split_mod_test (set of ints)

    Split default is "none": all modes see all samples (useful for debugging).
    """

    # CorrNet expects the feeder class to expose `collate_fn`
    collate_fn = staticmethod(dataloader_video.BaseFeeder.collate_fn)

    def __init__(
            self,
            # CorrNet-wired args (accepted for compatibility)
            prefix: Optional[str] = None,
            gloss_dict: Optional[Dict[str, List[int]]] = None,
            dataset: Optional[str] = None,
            kernel_size: Optional[Sequence[str]] = None,
            mode: str = "train",
            transform_mode: bool = True,
            frame_interval: int = 1,
            image_scale: int = 1,
            input_size: int = 224,

            # Your controls (via feeder_args)
            cache_root = Path("/mnt/c/Workplace/workspaces/ksl/cache/cached_npz_full"),
            use_cache: bool = True,
            w_start: int = 1,
            w_end: int = 30,
            angles: Sequence[str] = DEFAULT_ANGLES,

            # NEW: signer split (train=1..16, dev/test=17..18 by default)
            signers: Optional[Sequence[int]] = None,
            train_signers: Sequence[int] = tuple(range(1, 17)),
            dev_signers: Sequence[int] = (17, 18),

            # Optional split controls (kept for compatibility)
            split: str = "none",

            # Misc
            **_: Any,
    ):
        # Basic identity / compatibility
        self.prefix = prefix
        self.dataset = dataset
        self.mode = str(mode)
        self.transform_mode = "train" if transform_mode else "test"

        # CorrNet-style aug params (match BaseFeeder defaults)
        self.frame_interval = int(frame_interval)
        self.input_size = int(input_size)
        self.image_scale = image_scale

        # Load gloss_dict if not provided
        DEFAULT_GLOSS_DICT_PATH = Path(
            "/mnt/c/Workplace/workspaces/ksl/cache/preprocess\KSL\gloss_dict.npy"
        )
        if gloss_dict is None or len(gloss_dict) == 0:
            gloss_dict = np.load(DEFAULT_GLOSS_DICT_PATH, allow_pickle=True).item()

        self.gloss_dict = gloss_dict
        self.dict = gloss_dict  # CorrNet expects loader.dataset.dict

        # Cache controls
        self.cache_root = Path(cache_root)
        self.use_cache = bool(use_cache)

        # Data controls
        self.w_start = int(w_start)
        self.w_end = int(w_end)
        self.angles = tuple(angles)
        self.split = str(split)

        # Signer controls (IMPORTANT: signers affect *which samples exist*, not label IDs)
        mode_l = self.mode.lower()
        if signers is None:
            if mode_l in ("train", "training"):
                signers = train_signers
            elif mode_l in ("dev", "val", "valid", "validation", "test"):
                signers = dev_signers
            else:
                # safest default: include all you might have
                signers = tuple(sorted(set(train_signers) | set(dev_signers)))
        self.signers = tuple(int(p) for p in signers)

        # Build augmentation pipeline
        self.data_aug = self.transform()

        # CorrNet temporal padding expects a module-level kernel_sizes list
        self._kernel_sizes = list(kernel_size) if kernel_size is not None else [1]
        if kernel_size is not None:
            dataloader_video.kernel_sizes = list(kernel_size)

        # Build samples:
        all_samples = _iter_samples(self.w_start, self.w_end, self.signers, self.angles)
        self.samples = all_samples


    def __len__(self) -> int:
        return len(self.samples)

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def __getitem__(self, idx: int):
        from . import dataloader_video
        if getattr(dataloader_video, "kernel_sizes", None) is None:
            dataloader_video.kernel_sizes = self._kernel_sizes

        s = self.samples[idx]

        npz_path = self.cache_root / f"{s.stem}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing cache file: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as pack:
            video = pack["video"]
            label_id = int(pack["label_id"])

            label_str = None
            if "label_str" in pack.files:
                label_str = str(pack["label_str"])
            elif "label" in pack.files:
                label_str = str(pack["label"])

        if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"Bad cached video: dtype={video.dtype}, shape={video.shape}")

        label_list = [label_id]
        video, label_list = self.normalize(video, label_list, file_id=s.stem)

        label = torch.LongTensor(label_list)
        original_info = f"{s.stem}|{self.mode}/{s.stem}|0|{label_str}"
        return video, label, original_info


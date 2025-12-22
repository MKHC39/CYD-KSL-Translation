# cached_dataset.py
from pathlib import Path
from typing import Iterable, Tuple, Dict
import torch
from torch.utils.data import Dataset

from pydata import iter_samples, build_w_id_maps, ANGLES

class KSLCachedDataset(Dataset):
    def __init__(self, cache_dir: Path, w_start=1501, w_end=3000, angles: Iterable[str]=ANGLES):
        self.cache_dir = Path(cache_dir)
        self.samples = list(iter_samples(w_start, w_end, angles))
        self.w_to_id, self.id_to_w = build_w_id_maps(w_start, w_end)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        path = self.cache_dir / f"{s.stem}.pt"
        obj = torch.load(path, map_location="cpu")
        clip = obj["clip"]                       # (T,3,224,224)
        y = self.w_to_id[obj["w"]]               # contiguous id
        length = obj["length"]
        return clip, y, length, s.stem

def collate_pad_time(batch):
    clips, ys, lengths, stems = zip(*batch)
    B = len(clips)
    T_max = max(lengths)
    C, H, W = clips[0].shape[1:]
    out = torch.zeros((B, T_max, C, H, W), dtype=clips[0].dtype)
    for i, clip in enumerate(clips):
        out[i, : clip.shape[0]] = clip
    return out, torch.tensor(ys), torch.tensor(lengths), stems

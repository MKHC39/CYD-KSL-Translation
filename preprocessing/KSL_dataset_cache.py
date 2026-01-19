# cache_preprocess.py (drop-in replacement helpers)
# - Builds caches for multiple splits (train/dev/test) in one run
# - Replicates CorrNet's "loop over modes, write per-mode info, write one gloss_dict" pattern :contentReference[oaicite:0]{index=0}
#
# Key behaviour:
# - Different angles/signers do NOT create new classes: IDs are keyed ONLY by label_str.
# - Shared gloss_dict across splits (train->dev consistent IDs).
# - You can choose whether gloss_dict is built from train-only (recommended) or from all splits.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
import os
import argparse
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Optional

import numpy as np
import torch

from preprocessing.preprocess_clip import preprocess_stem

DEFAULT_ANGLES: Tuple[str, ...] = ("D", "F", "L", "R", "U")

_STEM_RE = re.compile(r"^NIA_SL_WORD(?P<w>\d{4})_REAL(?P<p>\d{2})_(?P<a>[DFLRU])$")


def parse_stem(stem: str) -> tuple[int, int, str]:
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Bad stem format: {stem}")
    return int(m.group("w")), int(m.group("p")), m.group("a")


@dataclass(frozen=True)
class CacheItem:
    stem: str
    w: int | None
    p: int | None
    angle: str


def iter_stems(
    w_start: int,
    w_end: int,
    signers: Sequence[int],
    angles: Sequence[str],
) -> Iterable[CacheItem]:
    for p in signers:
        for w in range(w_start, w_end + 1):
            for a in angles:
                stem = f"NIA_SL_WORD{w:04d}_REAL{p:02d}_{a}"
                yield CacheItem(stem=stem, w=w, p=p, angle=a)


def clip_chw01_to_uint8_thwc(clip: torch.Tensor) -> np.ndarray:
    """
    clip: (T,3,H,W) float32 in [0,1]  (preprocess_stem(..., normalise_imagenet=False))
    returns: (T,H,W,3) uint8 in [0,255]
    """
    if clip.ndim != 4 or clip.shape[1] != 3:
        raise ValueError(f"Expected (T,3,H,W), got {tuple(clip.shape)}")
    thwc = clip.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    thwc = np.clip(np.rint(thwc * 255.0), 0, 255).astype(np.uint8)
    return thwc


def save_npz(
    out_path: Path,
    video_uint8_thwc: np.ndarray,
    label_id: int,
    label_str: str,
    kept_indices: List[int],
    meta: Dict[str, Any],
    label_ids: Optional[Sequence[int]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta, ensure_ascii=False)
    # IMPORTANT: write to a file-handle so NumPy does NOT append ".npz"
    with open(out_path, "wb") as f:
        payload: Dict[str, Any] = {
            "video": video_uint8_thwc,
            "label_id": np.int32(label_id),
            "label_str": np.array(label_str),
            "video_len": np.int32(int(video_uint8_thwc.shape[0])),
            "kept_indices": np.asarray(kept_indices, dtype=np.int32),
            "meta_json": np.array(meta_json),
        }
        if label_ids is not None:
            payload["label_ids"] = np.asarray(label_ids, dtype=np.int32)
        np.savez(
            f,  # <- uncompressed NPZ container (faster load than savez_compressed)
            **payload,
        )


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def _write_info_npy(path: Path, info_rows: List[dict]) -> None:
    """
    CorrNet-style info is typically a dict-of-dicts keyed by fileid.
    We'll store: {stem: {...}}.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    info_dict = {r["fileid"]: r for r in info_rows}
    np.save(path, info_dict)


def find_dir(root: Path, target_name: str) -> Path:
    root = root.expanduser().resolve()

    # Fast path: common layout where it's directly under root
    direct = root / target_name
    if direct.is_dir():
        return direct

    # Otherwise: search recursively for a directory named target_name
    matches = [p for p in root.rglob(target_name) if p.is_dir()]

    if not matches:
        raise FileNotFoundError(f"Couldn't find a '{target_name}' directory under: {root}")

    # If there are multiple, pick the shortest path (closest to root)
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


_ANGLE_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<angle>[A-Za-z])$")
_NIASLG1_STEM_RE = re.compile(r"^NIA_SL_G1_[A-Z]+(?P<w>\d{6})_.+_(?P<angle>[A-Za-z])$")


def extract_niaslg1_word_num(stem: str) -> Optional[int]:
    m = _NIASLG1_STEM_RE.match(stem)
    if not m:
        return None
    return int(m.group("w"))


def stem_to_json_name(stem: str) -> str:
    m = _ANGLE_SUFFIX_RE.match(stem)
    base = m.group("base") if m else stem
    return f"{base}.json"


def load_sentence_gloss_map(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns {json_name: {"sentence": str, "gloss": str}}
    """
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            json_name = str(obj.get("json", "")).strip()
            if not json_name:
                continue
            rows[json_name] = {
                "sentence": str(obj.get("sentence", "")).strip(),
                "gloss": str(obj.get("gloss", "")).strip(),
            }
    return rows


def tokenize_gloss(gloss_seq: Any) -> List[str]:
    if isinstance(gloss_seq, list):
        return [str(x).strip() for x in gloss_seq if str(x).strip()]
    if isinstance(gloss_seq, str):
        return [t for t in gloss_seq.split() if t]
    return []


def write_stm(path: Path, rows: List[Dict[str, str]], speaker: str = "1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for r in rows:
        fileid = r["fileid"]
        gloss_seq = r["gloss"]
        lines.append(f"{fileid} 1 {speaker} 0.0 1.79769e+308 {gloss_seq}")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_cache_multi_split(
    *,
    cache_root: Path,
    out_dir: Path,
    dataset: str = "NIASL2021",
    splits: Mapping[str, Sequence[int]],          # e.g. {"train": range(1,17), "dev": (17,18)}
    angles: Sequence[str] = DEFAULT_ANGLES,
    step: int = 1,
    margin_px: int = 40,
    overwrite: bool = True,
    vocab_source: str = "train",                  # "train" (recommended) or "all"
    strict_dev_labels: bool = True,               # if True, dev/test unseen labels -> error
    data_root: Path,
    sentence_gloss_jsonl: Optional[Path] = None,
    niaslg1_train_dir: str = "1.Training",
    niaslg1_dev_dir: str = "2.Validation",
    write_stm_files: bool = True,
    stm_prefix: str = "KSL-groundtruth",
) -> None:
    """
    Replicates CorrNet's 'per-mode outputs + one gloss_dict' preprocessing pattern. :contentReference[oaicite:1]{index=1}

    Outputs:
      out_dir/
        gloss_dict.npy
        train_manifest.jsonl, dev_manifest.jsonl, ...
        train_info.npy, dev_info.npy, ...
    Cache:
      cache_root/{stem}.npz   (shared across splits; stem uniqueness already includes signer+angle)

    Label IDs:
      - keyed ONLY by label_str (meta["label"])
      - 0 reserved for blank (CTC); label IDs start at 1
    """
    if dataset == "NIASL2021":
        w_start = int(input("w_start = ?"))
        w_end = int(input("w_end = ?"))
        sentence_map: Dict[str, Dict[str, str]] = {}
        split_roots: Dict[str, Path] = {}
    elif dataset == "NIASLG1":
        w_start = int(input("w_start = ?"))
        w_end = int(input("w_end = ?"))
        if sentence_gloss_jsonl is None:
            raise ValueError("sentence_gloss_jsonl is required for NIASLG1")
        sentence_map = load_sentence_gloss_map(sentence_gloss_jsonl)
        split_roots = {
            "train": find_dir(data_root, niaslg1_train_dir),
            "dev": find_dir(data_root, niaslg1_dev_dir),
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


    cache_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_to_id: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}
    next_id = 1

    def ensure_label(label_str: str, allow_new: bool) -> int:
        nonlocal next_id
        if label_str in label_to_id:
            return label_to_id[label_str]
        if not allow_new:
            raise RuntimeError(f"Unseen label in non-vocab split: {label_str!r}")
        label_to_id[label_str] = next_id
        label_counts[label_str] = 0
        next_id += 1
        return label_to_id[label_str]

    # Decide which splits are allowed to add new labels
    vocab_source_l = vocab_source.strip().lower()
    if vocab_source_l not in ("train", "all"):
        raise ValueError("vocab_source must be 'train' or 'all'")

    # Split outputs
    split_manifests: Dict[str, List[dict]] = {k: [] for k in splits.keys()}
    split_info: Dict[str, List[dict]] = {k: [] for k in splits.keys()}

    # Stats
    stats = {k: {"ok": 0, "fail": 0, "skipped": 0} for k in splits.keys()}

    # Process in a deterministic order: train first, then others (so IDs stabilise early)
    split_order = list(splits.keys())
    if "train" in split_order:
        split_order.remove("train")
        split_order.insert(0, "train")

    for split_name in split_order:
        signers = splits.get(split_name, [])
        allow_new_labels = (vocab_source_l == "all") or (split_name == "train")
        if split_name != "train" and strict_dev_labels:
            allow_new_labels = False

        if dataset == "NIASL2021":
            items: List[Any] = list(iter_stems(w_start, w_end, signers, angles))
        else:
            split_root = split_roots.get(split_name)
            if split_root is None:
                continue
            mp4_paths = sorted(split_root.rglob("*.mp4"))
            items = []
            for p in mp4_paths:
                stem = p.stem
                word_num = extract_niaslg1_word_num(stem)
                if word_num is None or not (w_start <= word_num <= w_end):
                    continue
                m = _ANGLE_SUFFIX_RE.match(stem)
                angle = m.group("angle") if m else ""
                items.append({"stem": stem, "video_path": p, "angle": angle, "w": word_num})

        for item in items:
            stem = item.stem if isinstance(item, CacheItem) else item["stem"]
            out_path = cache_root / f"{stem}.npy"
            if out_path.exists() and not overwrite:
                stats[split_name]["skipped"] += 1
                # Still record manifest/info if you want; usually skip to keep manifests aligned with cache
                continue

            try:
                label_ids: Optional[List[int]] = None
                sentence = ""
                if dataset == "NIASL2021":
                    clip01, kept_indices, meta = preprocess_stem(
                        stem,
                        step=step,
                        margin_px=margin_px,
                        normalise_imagenet=False,
                        data_root=data_root,
                        dataset=dataset,
                    )
                    label_str = meta.get("label")
                    if not isinstance(label_str, str) or not label_str:
                        raise RuntimeError(f"Missing/invalid label in meta for {stem}: {label_str!r}")
                    label_id = ensure_label(label_str, allow_new=allow_new_labels)
                    label_counts[label_str] += 1
                else:
                    video_path = item["video_path"]
                    json_name = stem_to_json_name(stem)
                    row = sentence_map.get(json_name)
                    if row is None:
                        raise RuntimeError(f"Missing sentence/gloss for {json_name}")
                    gloss_seq = row["gloss"]
                    sentence = row["sentence"]
                    tokens = tokenize_gloss(gloss_seq)
                    label_ids = []
                    for tok in tokens:
                        tok_id = ensure_label(tok, allow_new=allow_new_labels)
                        label_counts[tok] = label_counts.get(tok, 0) + 1
                        label_ids.append(tok_id)
                    label_str = " ".join(tokens)
                    label_id = 0

                    clip01, kept_indices, meta = preprocess_stem(
                        stem,
                        step=step,
                        margin_px=margin_px,
                        normalise_imagenet=False,
                        data_root=data_root,
                        dataset=dataset,
                        video_path=video_path,
                        start_s=0.0,
                        end_s=None,
                        label_override=label_str,
                        skip_keypoints=True,
                    )

                video_uint8 = clip_chw01_to_uint8_thwc(clip01)
                T = int(video_uint8.shape[0])

                save_npz(
                    out_path=out_path,
                    video_uint8_thwc=video_uint8,
                    label_id=label_id,
                    label_str=label_str,
                    kept_indices=kept_indices,
                    meta=meta,
                    label_ids=label_ids,
                )

                split_manifests[split_name].append(
                    {
                        "stem": stem,
                        "w": int(item.w) if isinstance(item, CacheItem) else item.get("w"),
                        "p": int(item.p) if isinstance(item, CacheItem) else None,
                        "angle": item.angle if isinstance(item, CacheItem) else item["angle"],
                        "npz": str(out_path),
                        "label": label_str,
                        "label_id": int(label_id),
                        "T": T,
                        "split": split_name,
                    }
                )

                split_info[split_name].append(
                    {
                        "fileid": stem,
                        "split": split_name,
                        "w": int(item.w) if isinstance(item, CacheItem) else item.get("w"),
                        "p": int(item.p) if isinstance(item, CacheItem) else None,
                        "angle": item.angle if isinstance(item, CacheItem) else item["angle"],
                        "label": label_str,
                        "label_id": int(label_id),
                        "label_ids": label_ids,
                        "sentence": sentence,
                        "num_frames": T,
                        "original_info": f"{stem}|{label_str}|{label_id}",
                        "npz": str(out_path),
                    }
                )

                stats[split_name]["ok"] += 1
                print(f"[{split_name.upper()} OK] {stem} label={label_str} id={label_id} T={T}")

            except Exception as e:
                stats[split_name]["fail"] += 1
                print(f"[{split_name.upper()} FAIL] {stem}: {e}")

    # Write shared gloss_dict.npy (CorrNet-style {label_str: [id, count]})
    gloss_dict: Dict[str, List[int]] = {
        lab: [int(label_to_id[lab]), int(label_counts[lab])]
        for lab in sorted(label_to_id.keys())
    }
    np.save(out_dir / "gloss_dict.npy", gloss_dict)

    # Write per-split manifest + info
    for split_name in split_order:
        _write_jsonl(out_dir / f"{split_name}_manifest.jsonl", split_manifests[split_name])
        _write_info_npy(out_dir / f"{split_name}_info.npy", split_info[split_name])

    if dataset == "NIASLG1" and write_stm_files:
        for split_name in split_order:
            rows = [
                {"fileid": r["fileid"], "gloss": r["label"]}
                for r in split_info[split_name]
            ]
            write_stm(out_dir / f"{stm_prefix}-{split_name}.stm", rows)

    # Summary
    print("\n[DONE]")
    print(f"  cache_root: {cache_root}")
    print(f"  out_dir:    {out_dir}")
    print(f"  labels:     {len(gloss_dict)} (ids 1..{max([v[0] for v in gloss_dict.values()] + [0])})")
    for split_name in split_order:
        s = stats[split_name]
        print(f"  {split_name:>5}: ok={s['ok']} fail={s['fail']} skipped_existing={s['skipped']}")
    print(f"  wrote: {out_dir / 'gloss_dict.npy'}")
    for split_name in split_order:
        print(f"  wrote: {out_dir / f'{split_name}_manifest.jsonl'}")
        print(f"  wrote: {out_dir / f'{split_name}_info.npy'}")

def p(s: str) -> Path:
    return Path(s).expanduser()

def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_root",
        type = p,
        default = Path(__file__).parent.resolve() / "cache",
        help = "Path to the cache root directory",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset: (NIASL2021, NIASLG1)",
    )

    parser.add_argument(
        "--data_root",
        type = p,
        default= Path("/home/harrison/Workplace/workspaces/수어 재난 데이터"),
        help = r"Base directory for dataset; .../수어 영상 for NIASL2021",
    )

    parser.add_argument(
        "--sentence_gloss_jsonl",
        type=p,
        default = Path.cwd() / "glossTL/training_sentence_gloss.jsonl",
        help="Path to glossTL/training_sentence_gloss.jsonl (required for NIASLG1)",
    )

    parser.add_argument(
        "--niaslg1_train_dir",
        type=str,
        default="1.Training",
        help="Directory name for NIASLG1 train split under data_root",
    )

    parser.add_argument(
        "--niaslg1_dev_dir",
        type=str,
        default="2.Validation",
        help="Directory name for NIASLG1 dev split under data_root",
    )

    return parser


if __name__ == "__main__":
    # ---- EDIT THESE ----
    args = argparser().parse_args()
    DATA_ROOT = args.data_root
    CACHE_ROOT = args.cache_root
    OUT_DIR    = args.cache_root / "preprocess" / "KSL"
    DATASET    = args.dataset
    SENTENCE_GLOSS_JSONL = args.sentence_gloss_jsonl
    NIASLG1_TRAIN_DIR = args.niaslg1_train_dir
    NIASLG1_DEV_DIR = args.niaslg1_dev_dir

    SPLITS = {
        "train": list(range(1, 17)),  # signers 1..16
        "dev": (17, 18),              # signers 17..18 (validation set)
        # "test": (...)               # if you add later
    }

    ANGLES  = DEFAULT_ANGLES

    STEP = 1
    MARGIN_PX = 40
    OVERWRITE = True

    # Recommended: build vocab from train only, and error if dev has unseen labels
    VOCAB_SOURCE = "train"          # "train" or "all"
    STRICT_DEV_LABELS = True
    # -------------------

    build_cache_multi_split(
        cache_root=CACHE_ROOT,
        out_dir=OUT_DIR,
        splits=SPLITS,
        dataset=DATASET,
        angles=ANGLES,
        step=STEP,
        margin_px=MARGIN_PX,
        overwrite=OVERWRITE,
        vocab_source=VOCAB_SOURCE,
        strict_dev_labels=STRICT_DEV_LABELS,
        data_root=DATA_ROOT,
        sentence_gloss_jsonl=SENTENCE_GLOSS_JSONL,
        niaslg1_train_dir=NIASLG1_TRAIN_DIR,
        niaslg1_dev_dir=NIASLG1_DEV_DIR,
    )

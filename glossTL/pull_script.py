from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List, Union
import pandas as pd

import re
import json

from pathlib import Path
import subprocess

from pathmagic import smart_path as sp


def wsl_path(p: str) -> Path:
    """
    Accepts either:
      - Windows path copied from Explorer:  C:\\Users\\... or D:\\...
      - WSL/Linux path: /mnt/c/... etc
      - WSL UNC path: \\\\wsl.localhost\\Distro\\home\\... (from Explorer)
    Returns a valid WSL Path (requires WSL's `wslpath` for Windows inputs).
    """
    s = p.strip().strip('"').strip("'")
    if not s:
        raise ValueError("Empty path")
    # Handle WSL UNC path copied from Windows Explorer
    # Example: \\wsl.localhost\Ubuntu\home\user\project
    if r"wsl.localhost" in s.lower():
        parts = s.split("\\")
        # ["", "", "wsl.localhost", "Ubuntu", "home", "user", ...]
        if len(parts) >= 5:
            s = "/" + "/".join(parts[4:])  # drop \\wsl.localhost\Distro
            return Path(s).expanduser().resolve()

    # Heuristic: Windows drive path like C:\...
    if len(s) >= 3 and s[1] == ":" and (s[2] == "\\" or s[2] == "/"):
        s = subprocess.check_output(["wslpath", "-u", s], text=True).strip()

    return Path(s).expanduser().resolve()


def parse_df(df: pd.DataFrame, source: str ="<df>", fatal_codes: set[str] | None = None
             ) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """
    df read with header=None.
    Columns: 0 = class/label, 1 = field name, 2.. = payload.

    Returns:
      { json_filename: { "sentence": str|None, "segments": [(label, gloss, start, end, row, column), ...] } }
    """
    if fatal_codes is None:
        fatal_codes = {
            "MISSING_JSON",
            "EMPTY_SENTENCE",
            "MISSING_SENTENCE",
            "MISSING_END_ROW",
            "MISSING_END_VALUE",
            "END_BEFORE_START",
            "NONNUMERIC_START",
            "NONNUMERIC_END",
        }


    out: dict[str, dict[str, Any]] = {}
    drop_rows: list[dict] = []

    current_json: Optional[str] = None
    current_sentence: Optional[str] = None
    current_segments: list[tuple[str, Optional[str], float, float, int, int]] = []
    current_invalid = False
    current_fatal_reasons: set[str] = set()
    current_first_fatal: dict | None = None

    n_rows = len(df)
    n_cols = df.shape[1]

    def is_field(x: Any, name: str) -> bool:
        return isinstance(x, str) and x.strip() == name

    def mark_fatal(code: str, row: Optional[int] = None, col: Optional[int] = None, msg: str = ""):
        nonlocal current_invalid, current_first_fatal
        if code in fatal_codes:
            current_invalid = True
            current_fatal_reasons.add(code)
            if current_first_fatal is None:
                current_first_fatal = {
                    "source" : source,
                    "json" : current_json,
                    "row" : row,
                    "col" : col,
                    "code" : code,
                    "message" : msg
                }


    def flush_current(end_row: Optional[int] = None) -> None:
        nonlocal current_json, current_sentence, current_segments, current_invalid, current_fatal_reasons, current_first_fatal
        if not current_json:
            if current_invalid:
                drop_rows.append({
                    "source": source,
                    "json": None,
                    "sentence": None if current_sentence is None else str(current_sentence),
                    "fatal_codes": ",".join(sorted(current_fatal_reasons)),
                    "first_code": current_first_fatal["code"] if current_first_fatal else None,
                    "row": current_first_fatal["row"] if current_first_fatal else end_row,
                    "col": current_first_fatal["col"] if current_first_fatal else None,
                    "message": current_first_fatal["message"] if current_first_fatal else None,
                })
                return
        if current_invalid:
            drop_rows.append({
                "source": source,
                "json": None,
                "sentence": None if current_sentence is None else str(current_sentence),
                "fatal_codes": ",".join(sorted(current_fatal_reasons)),
                "first_code": current_first_fatal["code"] if current_first_fatal else None,
                "row": current_first_fatal["row"] if current_first_fatal else end_row,
                "col": current_first_fatal["col"] if current_first_fatal else None,
                "message": current_first_fatal["message"] if current_first_fatal else None,
            })
            return

        if not current_segments:
            return

        current_segments.sort(key=lambda t: t[2])  # sort by start
        out[str(current_json)] = {
            "sentence": None if current_sentence is None or pd.isna(current_sentence) else str(current_sentence),
            "segments": current_segments,
        }

    row = 0

    while row < n_rows:
        label = df.iat[row, 0] if n_cols > 0 else None
        info = df.iat[row, 1] if n_cols > 1 else None

        # ---- Record boundary / metadata start ----
        if label == "Information" and is_field(info, "File name :"):
            # New record begins: flush previous one
            flush_current(end_row = row)

            # Start new record
            current_json = None
            current_sentence = None
            current_segments = []
            current_invalid = False
            current_fatal_reasons = set()
            current_first_fatal = None

            if n_cols > 2 and pd.notna(df.iat[row, 2]):
                current_json = str(df.iat[row, 2])
            else:
                mark_fatal("MISSING_JSON", row=row, col=2, msg="Missing JSON")

            # Find Korean sentence nearby (often next row, but scan a small window)
            found_sentence = False
            for i in range(row, min(row + 15, n_rows)):
                f = df.iat[i, 1] if n_cols > 1 else None
                if is_field(f, "Korean sentence :"):
                    found_sentence = True
                    val = df.iat[i,2] if n_cols > 2 else None
                    if pd.isna(val):
                        mark_fatal("EMPTY_SENTENCE", row=i, col=2, msg="Korean sentence is empty/NaN")
                    else:
                        current_sentence = str(val)
                    break
            if not found_sentence:
                mark_fatal("MISSING_SENTENCE", row=row, col=None, msg="No Korean sentence field near Information block")

            row += 1
            continue

        # ---- Segment parsing: anchor on start(s) rows ----
        if is_field(info,"start(s) :"):
            # Define the three rows (gloss above, starts here, ends below)
            starts_row = df.iloc[row, :]
            ends_row = None
            if row + 1 < n_rows and is_field(df.iat[row + 1, 1] if n_cols > 1 else None, "end(s) :"):
                ends_row = df.iloc[row + 1, :]
            else:
                mark_fatal("MISSING_END_ROW", row=row, col=1, msg="start(s) not followed by end(s)")
                row += 1
                continue

            gloss_row = None
            if row - 1 >= 0 and is_field(df.iat[row - 1, 1] if n_cols > 1 else None, "gloss_id :"):
                gloss_row = df.iloc[row - 1, :]

            # Determine label (prefer gloss row's col0)
            label = None
            if gloss_row is not None and pd.notna(gloss_row.iat[0]):
                label = str(gloss_row.iat[0])

            # Non-reuse: consume end cells within this ends_row
            used_end_cols: set[int] = set()

            # For every non-NaN start in columns >=2
            for j in range(2, n_cols):
                sv = starts_row.iat[j]
                if pd.isna(sv):
                    continue
                try:
                    start = float(sv)
                except Exception:
                    mark_fatal("NONNUMERIC_START", row=row, col=j, msg=f"Start not numeric: {sv}")
                    continue

                gloss = None
                if gloss_row is not None:
                    gv = gloss_row.iat[j]
                    if pd.notna(gv):
                        gloss = str(gv)

                if gloss is None:
                    continue

                # Find end: prefer same column; else scan right to next non-NaN number
                end = None
                end_col = None
                for jj in range(j, n_cols):
                    if jj in used_end_cols:
                        continue
                    ev = ends_row.iat[jj]
                    if pd.notna(ev):
                        try:
                            end = float(ev)
                            end_col = jj
                        except Exception:
                            mark_fatal("NONNUMERIC_END", row=row + 1, col=jj, msg=f"End not numeric: {ev}")
                            end = None
                            end_col = None
                        break

                if end is None or end_col is None:
                    mark_fatal("MISSING_END_VALUE", row=row + 1, col=j, msg="No end time found to the right")
                    continue

                used_end_cols.add(end_col)

                if end < start:
                    mark_fatal("END_BEFORE_START", row=row, col=j, msg=f"End {end} < Start {start}")
                    continue

                current_segments.append((label, gloss, start, end, row, end_col))

            # Skip past the paired end row as well
            row += 2
            continue

        row += 1

    # Flush the last record
    flush_current(end_row = n_rows-1)

    drops_df = pd.DataFrame(drop_rows, columns = [
        "source", "json", "sentence", "fatal_codes", "first_code", "row", "col", "message"
    ])

    return out, drops_df


def dict_to_dataframes(data: Dict[str, Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input:
      data[json] = {"sentence": str|None, "segments": [(label, gloss, start, end), ...]}

    Output:
      meta_df:    entry_id, sentence, json
      segments_df: entry_id, label, gloss, start, end, ord
    """
    meta_rows: List[Tuple[int, Optional[str], str]] = []
    seg_rows: List[Tuple[int, str, Optional[str], float, float, int, int, int]] = []

    entry_id = 0
    for json_name, rec in data.items():
        sentence = rec.get("sentence", None)
        segments = rec.get("segments", []) or []

        meta_rows.append((entry_id, sentence, json_name))

        # ensure sorted by start, then add an ordinal for stable sequencing
        segments_sorted = sorted(segments, key=lambda t: t[2])
        for k, (label, gloss, start, end, row, end_col) in enumerate(segments_sorted):
            seg_rows.append((entry_id, label, gloss, float(start), float(end), row, end_col, k))

        entry_id += 1

    meta_df = pd.DataFrame(meta_rows, columns=["entry_id", "sentence", "json"])
    segments_df = pd.DataFrame(seg_rows, columns=["entry_id", "label", "gloss", "start", "end", "row", "column", "ord"])

    # dtypes (memory-friendly)
    meta_df["entry_id"] = meta_df["entry_id"].astype("int32")
    meta_df["sentence"] = meta_df["sentence"].astype("string")
    meta_df["json"] = meta_df["json"].astype("string")

    segments_df["entry_id"] = segments_df["entry_id"].astype("int32")
    segments_df["label"] = segments_df["label"].astype("category")
    segments_df["gloss"] = segments_df["gloss"].astype("category")
    segments_df["start"] = segments_df["start"].astype("float32")
    segments_df["end"] = segments_df["end"].astype("float32")
    segments_df["ord"] = segments_df["ord"].astype("int16")

    return meta_df, segments_df


def folder_df(folder: Path):
    files = folder.glob("*.xlsx")
    all_data = {}
    all_drops = []

    for f in files:
        df = pd.read_excel(f, header=None, engine="openpyxl")
        data, drops = parse_df(df, source=str(f))
        all_data.update(data)
        if not drops.empty:
            all_drops.append(drops)

    if all_drops:
        drops_df = pd.concat(all_drops, ignore_index=True)
    else:
        drops_df = pd.DataFrame()

    meta_df, segments_df = dict_to_dataframes(all_data)

    return meta_df, segments_df, drops_df


_TRAILING_DIGITS_RE = re.compile(r"\d+$")
_TRAILING_HASHES_RE = re.compile(r"[#'@]+$")

def normalise_gloss_token(token: str) -> str:
    token = str(token).strip()
    if not token:
        return token

    # Remove trailing '#' markers (e.g., "오늘1#" -> "오늘1")
    token = _TRAILING_HASHES_RE.sub("", token).strip()

    # Keep pure numbers (e.g., phone number chunks)
    if token.isdigit():
        return token

    # Remove trailing occurrence index digits (e.g., "오늘12" -> "오늘")
    token = _TRAILING_DIGITS_RE.sub("", token).strip()
    return token


def sent_gloss(
        meta_df: pd.DataFrame,
        segments_df: pd.DataFrame,
        *,
        gloss_as_list: bool = False,
        sep: str = " ",
        dropna_gloss: bool = True,
        keep_no_segments: bool = False,
) -> Dict[int, Tuple[str, Union[List[str], str], str, str]]:
    """
    Build: {entry_id: (sentence, gloss_seq, json, src)}

    - sentence comes from meta_df
    - gloss_seq comes from segments_df rows matching entry_id, sorted by ord
    - gloss_seq is a list by default (gloss_as_list=True); otherwise a sep-joined string

    Expected columns:
      meta_df:     entry_id, sentence, json
      segments_df: entry_id, label, gloss, start, end, ord
    """
    required_meta = {"entry_id", "sentence", "json"}
    required_seg = {"entry_id", "gloss", "ord", "start", "end"}

    missing_meta = required_meta - set(meta_df.columns)
    missing_seg = required_seg - set(segments_df.columns)
    if missing_meta:
        raise ValueError(f"meta_df missing required columns: {sorted(missing_meta)}")
    if missing_seg:
        raise ValueError(f"segments_df missing required columns: {sorted(missing_seg)}")

    m = meta_df[["entry_id", "sentence", "json", "source_folder"]].copy()
    s = segments_df[["entry_id", "gloss", "ord", "start", "end"]].copy()

    # Normalise types
    m["entry_id"] = m["entry_id"].astype(int)
    s["entry_id"] = s["entry_id"].astype(int)
    s["ord"] = s["ord"].astype(int)

    # Clean gloss tokens
    if dropna_gloss:
        s = s.dropna(subset=["gloss"])
    s["gloss"] = s["gloss"].astype(str).str.strip()
    s["gloss"] = s["gloss"].map(normalise_gloss_token).str.strip()
    if dropna_gloss:
        s = s[s["gloss"] != ""]

    # Sort by intended order and aggregate tokens per entry_id
    s = s.sort_values(["entry_id", "ord"], kind="mergesort")

    def _overlaps(a_start, a_end, b_start, b_end) -> bool:
        if pd.isna(a_start) or pd.isna(a_end) or pd.isna(b_start) or pd.isna(b_end):
            return False
        return max(a_start, b_start) <= min(a_end, b_end)

    def _merge_gloss_runs(group: pd.DataFrame) -> List[str]:
        tokens: List[str] = []
        prev_gloss = None
        prev_start = None
        prev_end = None

        for gloss, start, end in zip(group["gloss"], group["start"], group["end"]):
            if prev_gloss is not None and gloss == prev_gloss and _overlaps(prev_start, prev_end, start, end):
                if not pd.isna(end) and (pd.isna(prev_end) or end > prev_end):
                    prev_end = end
                continue

            tokens.append(gloss)
            prev_gloss = gloss
            prev_start = start
            prev_end = end

        return tokens

    gloss_by_id = s.groupby("entry_id", sort=False).apply(_merge_gloss_runs)

    out: Dict[int, Tuple[str, Union[List[str], str],str, str]] = {}

    for entry_id, sentence, json_name ,src in zip(m["entry_id"], m["sentence"],m["json"], m["source_folder"]):
        tokens = gloss_by_id.get(entry_id, [])
        if (not keep_no_segments) and (len(tokens) == 0):
            continue
        gloss_seq: Union[List[str], str] = tokens if gloss_as_list else sep.join(tokens)
        out[int(entry_id)] = (sentence, gloss_seq, json_name, str(src))

    return out


def iter_xlsx_folders(root: Path):
    """
    Yield each unique folder under `root` that contains at least one .xlsx file.
    """
    folders = {p.parent for p in root.rglob("*.xlsx")}
    for folder in sorted(folders):
        yield folder


def merge_folder_outputs(root: Path, *, verbose: bool = True):
    meta_parts = []
    seg_parts = []
    drop_parts = []

    next_id = 0
    folders_done = 0

    for folder in iter_xlsx_folders(root):
        meta_df, segments_df, drops_df = folder_df(folder)

        # Normalise types
        meta_df = meta_df.copy()
        segments_df = segments_df.copy()
        meta_df["entry_id"] = meta_df["entry_id"].astype(int)
        segments_df["entry_id"] = segments_df["entry_id"].astype(int)

        # Map local entry_id -> global entry_id
        local_ids = meta_df["entry_id"].unique()
        id_map = {int(lid): int(next_id + i) for i, lid in enumerate(sorted(local_ids))}
        next_id += len(id_map)

        meta_df["entry_id"] = meta_df["entry_id"].map(id_map)
        segments_df["entry_id"] = segments_df["entry_id"].map(id_map)

        # Optional provenance
        meta_df["source_folder"] = str(folder)
        segments_df["source_folder"] = str(folder)

        meta_parts.append(meta_df)
        seg_parts.append(segments_df)

        if drops_df is not None and not drops_df.empty:
            drops_df = drops_df.copy()
            drops_df["source_folder"] = str(folder)
            drop_parts.append(drops_df)

        folders_done += 1
        if verbose:
            xlsx_count = len(list(folder.glob("*.xlsx")))
            print(
                f"[{folders_done:04d}] done: {folder} | "
                f"xlsx={xlsx_count} | "
                f"meta_rows={len(meta_df)} | "
                f"seg_rows={len(segments_df)} | "
                f"drops_rows={(len(drops_df) if drops_df is not None else 0)} | "
                f"global_ids_now={next_id}"
            )

    meta_all = pd.concat(meta_parts, ignore_index=True) if meta_parts else pd.DataFrame()
    seg_all = pd.concat(seg_parts, ignore_index=True) if seg_parts else pd.DataFrame()
    drops_all = pd.concat(drop_parts, ignore_index=True) if drop_parts else pd.DataFrame()

    if verbose:
        print(
            f"\nFinished. folders={folders_done} | "
            f"meta_all={len(meta_all)} | seg_all={len(seg_all)} | drops_all={len(drops_all)}"
        )

    return meta_all, seg_all, drops_all

def save_jsonl(
    data: dict[int, tuple[str, str, str, str]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for entry_id, (sentence, gloss_seq, json_name ,source_folder) in data.items():
            obj = {
                "entry_id": entry_id,
                "sentence": sentence,
                "gloss": gloss_seq,
                "json": json_name,
                "source_folder": source_folder
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _extract_json_segments(script: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if not isinstance(script, dict):
        return segments

    for value in script.values():
        if not value:
            continue
        if isinstance(value, list):
            iterable = value
        elif isinstance(value, dict):
            iterable = value.values()
        else:
            continue

        for segment in iterable:
            if not isinstance(segment, dict):
                continue
            gloss = segment.get("descriptor")
            if gloss is None or (isinstance(gloss, str) and not gloss.strip()):
                gloss = segment.get("gloss_id")

            start = segment.get("start")
            end = segment.get("end")
            try:
                start = float(start)
                end = float(end)
            except (TypeError, ValueError):
                continue

            segments.append({"gloss": gloss, "start": start, "end": end})

    return segments


def parse_json_data(data: dict[str, Any]) -> tuple[Optional[str], list[dict[str, Any]]]:
    sentence = data.get("sentence")
    if sentence is None:
        sentence = data.get("korean_text")

    segments = []
    segments.extend(_extract_json_segments(data.get("nms_script", {})))
    segments.extend(_extract_json_segments(data.get("sign_script", {})))
    segments.sort(key=lambda x: x["start"])

    return sentence, segments


def json_folder_df(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows: list[tuple[int, Optional[str], str]] = []
    seg_rows: list[tuple[int, Optional[str], Optional[str], float, float, int]] = []

    entry_id = 0
    for f in sorted(folder.glob("*.json")):
        with f.open("r", encoding="utf-8") as file:
            data = json.load(file)

        sentence, segments = parse_json_data(data)
        meta_rows.append((entry_id, sentence, f.name))

        for ord_idx, segment in enumerate(segments):
            seg_rows.append(
                (entry_id, None, segment.get("gloss"), segment["start"], segment["end"], ord_idx)
            )

        entry_id += 1

    meta_df = pd.DataFrame(meta_rows, columns=["entry_id", "sentence", "json"])
    segments_df = pd.DataFrame(
        seg_rows, columns=["entry_id", "label", "gloss", "start", "end", "ord"]
    )

    return meta_df, segments_df


def iter_json_folders(root: Path):
    folders = {p.parent for p in root.rglob("*.json")}
    for folder in sorted(folders):
        yield folder


def merge_json_outputs(root: Path, *, verbose: bool = True):
    meta_parts = []
    seg_parts = []

    next_id = 0
    folders_done = 0

    for folder in iter_json_folders(root):
        meta_df, segments_df = json_folder_df(folder)

        if meta_df.empty and segments_df.empty:
            continue

        meta_df = meta_df.copy()
        segments_df = segments_df.copy()
        meta_df["entry_id"] = meta_df["entry_id"].astype(int)
        segments_df["entry_id"] = segments_df["entry_id"].astype(int)

        local_ids = meta_df["entry_id"].unique()
        id_map = {int(lid): int(next_id + i) for i, lid in enumerate(sorted(local_ids))}
        next_id += len(id_map)

        meta_df["entry_id"] = meta_df["entry_id"].map(id_map)
        segments_df["entry_id"] = segments_df["entry_id"].map(id_map)

        meta_df["source_folder"] = str(folder)
        segments_df["source_folder"] = str(folder)

        meta_parts.append(meta_df)
        seg_parts.append(segments_df)

        folders_done += 1
        if verbose:
            json_count = len(list(folder.glob("*.json")))
            print(
                f"[{folders_done:04d}] done: {folder} | "
                f"json={json_count} | "
                f"meta_rows={len(meta_df)} | "
                f"seg_rows={len(segments_df)} | "
                f"global_ids_now={next_id}"
            )

    meta_all = pd.concat(meta_parts, ignore_index=True) if meta_parts else pd.DataFrame()
    seg_all = pd.concat(seg_parts, ignore_index=True) if seg_parts else pd.DataFrame()

    if verbose:
        print(
            f"\nFinished. folders={folders_done} | "
            f"meta_all={len(meta_all)} | seg_all={len(seg_all)}"
        )

    return meta_all, seg_all



if __name__ == "__main__":
    win_path = input("Please enter your Windows path: ")
    root = sp(win_path)

    mode = input("Mode [xlsx/json] (default: xlsx): ").strip().lower() or "xlsx"

    if mode == "json":
        meta_all, seg_all = merge_json_outputs(root)
        merged_dict = sent_gloss(meta_all, seg_all)

        base = Path(__file__).resolve().parent
        out_path = base / "training_sentence_gloss_json.jsonl"
        serialisable = {str(k): [v[0], v[1]] for k, v in merged_dict.items()}

        save_jsonl(merged_dict, out_path)

        print(f"meta_all rows: {len(meta_all)}")
        print(f"seg_all rows:  {len(seg_all)}")
        print(f"Saved {len(serialisable)} entries to: {out_path}")
    else:
        meta_all, seg_all, drops_all = merge_folder_outputs(root)
        merged_dict = sent_gloss(meta_all, seg_all)

        base = Path(__file__).resolve().parent
        out_path = base / "training_sentence_gloss.jsonl"
        serialisable = {str(k): [v[0], v[1]] for k, v in merged_dict.items()}

        save_jsonl(merged_dict, out_path)

        print(f"meta_all rows: {len(meta_all)}")
        print(f"seg_all rows:  {len(seg_all)}")
        print(f"drops_all rows:{len(drops_all)}")
        print(f"Saved {len(serialisable)} entries to: {out_path}")

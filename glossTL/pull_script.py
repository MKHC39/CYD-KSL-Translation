from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import pandas as pd

from pathlib import Path
import subprocess


def wsl_path(p: str) -> Path:
    """
    Accepts either:
      - Windows path copied from Explorer:  C:\\Users\\... or D:\\...
      - WSL/Linux path: /mnt/c/... etc
    Returns a valid WSL Path (requires WSL's `wslpath` for Windows inputs).
    """
    s = p.strip().strip('"').strip("'")
    if not s:
        raise ValueError("Empty path")

    # Heuristic: Windows drive path like C:\...
    if len(s) >= 3 and s[1] == ":" and (s[2] == "\\" or s[2] == "/"):
        s = subprocess.check_output(["wslpath", "-u", s], text=True).strip()

    return Path(s)


def parse_df(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    df read with header=None.
    Columns: 0 = class/label, 1 = field name, 2.. = payload.

    Returns:
      { json_filename: { "sentence": str|None, "segments": [(label, gloss, start, end), ...] } }
    """
    out: dict[str, dict[str, Any]] = {}

    def flush_current(json_name: Optional[str],
                      sentence: Optional[str],
                      segments: list[tuple[str, Optional[str], float, float]]) -> None:
        if not json_name:
            return
        segments.sort(key=lambda t: t[2])  # sort by start
        out[str(json_name)] = {
            "sentence": None if sentence is None or pd.isna(sentence) else str(sentence),
            "segments": segments,
        }

    current_json: Optional[str] = None
    current_sentence: Optional[str] = None
    current_segments: list[tuple[str, Optional[str], float, float]] = []

    row = 0
    n_rows = len(df)
    n_cols = df.shape[1]

    while row < n_rows:
        label = df.iat[row, 0] if n_cols > 0 else None
        info = df.iat[row, 1] if n_cols > 1 else None

        # ---- Record boundary / metadata start ----
        if label == "Information" and isinstance(info, str) and info.strip() == "File name :":
            # New record begins: flush previous one
            flush_current(current_json, current_sentence, current_segments)

            # Start new record
            current_json = df.iat[row, 2] if n_cols > 2 else None
            current_sentence = None
            current_segments = []

            # Find Korean sentence nearby (often next row, but scan a small window)
            for i in range(row, min(row + 15, n_rows)):
                f = df.iat[i, 1] if n_cols > 1 else None
                if isinstance(f, str) and f.strip() == "Korean sentence :":
                    current_sentence = df.iat[i, 2] if n_cols > 2 else None
                    break

            row += 1
            continue

        # ---- Segment parsing: anchor on start(s) rows ----
        if isinstance(info, str) and info.strip() == "start(s) :":
            # Define the three rows (gloss above, starts here, ends below)
            starts_row = df.iloc[row, :]
            ends_row = None
            if row + 1 < n_rows:
                info_below = df.iat[row + 1, 1] if n_cols > 1 else None
                if isinstance(info_below, str) and info_below.strip() == "end(s) :":
                    ends_row = df.iloc[row + 1, :]

            gloss_row = None
            if row - 1 >= 0:
                info_above = df.iat[row - 1, 1] if n_cols > 1 else None
                if isinstance(info_above, str) and info_above.strip() == "gloss_id :":
                    gloss_row = df.iloc[row - 1, :]

            # Determine label (prefer gloss row's col0)
            label = None
            if gloss_row is not None and pd.notna(gloss_row.iat[0]):
                label = str(gloss_row.iat[0])
            elif pd.notna(starts_row.iat[0]):
                label = str(starts_row.iat[0])
            else:
                label = "UNKNOWN"

            if ends_row is None:
                # Can't pair ends; skip this start(s) row safely
                raise Exception("No end row found")

            # Non-reuse: consume end cells within this ends_row
            used_end_cols: set[int] = set()

            # For every non-NaN start in columns >=2
            for j in range(2, n_cols):
                sv = starts_row.iat[j]
                if pd.isna(sv):
                    continue
                start = float(sv)

                gloss: Optional[str] = None
                if gloss_row is not None:
                    gv = gloss_row.iat[j]
                    if pd.notna(gv):
                        gloss = str(gv)

                # Find end: prefer same column; else scan right to next non-NaN number
                end: Optional[float] = None
                end_col: Optional[int] = None
                for jj in range(j, n_cols):
                    if jj in used_end_cols:
                        continue
                    ev = ends_row.iat[jj]
                    if pd.notna(ev):
                        end = float(ev)
                        end_col = jj
                        break

                if end is None or end_col is None:
                    raise Exception(f"No end value found at {row+1},{end_col}")
                    continue

                used_end_cols.add(end_col)
                current_segments.append((label, gloss, start, end))

            # Skip past the paired end row as well
            row += 2
            continue

        row += 1

    # Flush the last record
    flush_current(current_json, current_sentence, current_segments)
    return out


def dict_to_dataframes(data: Dict[str, Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input:
      data[json] = {"sentence": str|None, "segments": [(label, gloss, start, end), ...]}

    Output:
      meta_df:    entry_id, sentence, json
      segments_df: entry_id, label, gloss, start, end, ord
    """
    meta_rows: List[Tuple[int, Optional[str], str]] = []
    seg_rows: List[Tuple[int, str, Optional[str], float, float, int]] = []

    entry_id = 0
    for json_name, rec in data.items():
        sentence = rec.get("sentence", None)
        segments = rec.get("segments", []) or []

        meta_rows.append((entry_id, sentence, json_name))

        # ensure sorted by start, then add an ordinal for stable sequencing
        segments_sorted = sorted(segments, key=lambda t: t[2])
        for k, (label, gloss, start, end) in enumerate(segments_sorted):
            seg_rows.append((entry_id, label, gloss, float(start), float(end), k))

        entry_id += 1

    meta_df = pd.DataFrame(meta_rows, columns=["entry_id", "sentence", "json"])
    segments_df = pd.DataFrame(seg_rows, columns=["entry_id", "label", "gloss", "start", "end", "ord"])

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


def main(folder: Path):
    files = folder.glob("*.xlsx")
    all_data = {}

    for f in files:
        df = pd.read_excel(f, header=None, engine="openpyxl")
        all_data.update(parse_df(df))

    meta_df, segments_df = dict_to_dataframes(all_data)
    # meta_df.head()
    # segments_df.head()

    return meta_df, segments_df

if __name__ == "__main__":
    win_path = input("Please enter your Windows path: ")
    folder = wsl_path(win_path)
    main(folder)
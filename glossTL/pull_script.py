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


def parse_df(df: pd.DataFrame, source: str ="<df>", fatal_codes: set[str] | None = None
             ) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """
    df read with header=None.
    Columns: 0 = class/label, 1 = field name, 2.. = payload.

    Returns:
      { json_filename: { "sentence": str|None, "segments": [(label, gloss, start, end), ...] } }
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
    current_segments: list[tuple[str, Optional[str], float, float]] = []
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

                current_segments.append((label, gloss, start, end))

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
    all_drops = []

    for f in files:
        df = pd.read_excel(f, header=None, engine="openpyxl")
        data, drops = parse_df(df, source=str(f))
        all_data.update(data)
        if not drops.empty:
            all_drops.append(drops)

    drops_df = pd.concat(all_drops, ignore_index=True)

    meta_df, segments_df = dict_to_dataframes(all_data)

    return meta_df, segments_df, drops_df

if __name__ == "__main__":
    win_path = input("Please enter your Windows path: ")
    folder = wsl_path(win_path)
    main(folder)
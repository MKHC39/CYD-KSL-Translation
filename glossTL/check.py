from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, List, Dict, Tuple

import pandas as pd


def _is_field(x: Any, name: str) -> bool:
    return isinstance(x, str) and x.strip() == name


def _safe_float(x: Any) -> Optional[float]:
    if pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def validate_workbook_df(df: pd.DataFrame, source: str = "<df>") -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates one workbook loaded with header=None.

    Logs structural/value errors but DOES NOT treat missing gloss tokens/labels as errors.
    """
    errs: List[Dict[str, Any]] = []

    n_rows = len(df)
    n_cols = df.shape[1]

    records_seen = 0
    records_with_sentence = 0
    triplets_seen = 0
    segments_seen = 0

    current_json: Optional[str] = None
    current_has_sentence = False

    def log(code: str, row: int, col: Optional[int], msg: str, extra: Dict[str, Any] | None = None):
        d = {
            "source": source,
            "json": current_json,
            "row": row,
            "col": col,
            "code": code,
            "message": msg,
        }
        if extra:
            d.update(extra)
        errs.append(d)

    i = 0
    while i < n_rows:
        c0 = df.iat[i, 0] if n_cols > 0 else None
        c1 = df.iat[i, 1] if n_cols > 1 else None

        # --- Record boundary ---
        if c0 == "Information" and _is_field(c1, "File name :"):
            # Close previous record (sentence missing is usually worth logging)
            if current_json is not None and not current_has_sentence:
                log("MISSING_SENTENCE", i, None, "Record ended without a Korean sentence")

            records_seen += 1
            current_json = None if n_cols <= 2 or pd.isna(df.iat[i, 2]) else str(df.iat[i, 2])
            current_has_sentence = False

            if current_json is None:
                log("MISSING_JSON", i, 2, "Information block has missing File name value")

            # Look ahead for sentence
            for k in range(i, min(i + 15, n_rows)):
                f = df.iat[k, 1] if n_cols > 1 else None
                if _is_field(f, "Korean sentence :"):
                    s = df.iat[k, 2] if n_cols > 2 else None
                    if pd.isna(s):
                        log("EMPTY_SENTENCE", k, 2, "Korean sentence field exists but value is NaN/empty")
                    else:
                        current_has_sentence = True
                        records_with_sentence += 1
                    break

            i += 1
            continue

        # --- Triplet anchored on start(s) ---
        if _is_field(c1, "start(s) :"):
            triplets_seen += 1

            starts_row = df.iloc[i, :]
            ends_row = None
            if i + 1 < n_rows and _is_field(df.iat[i + 1, 1] if n_cols > 1 else None, "end(s) :"):
                ends_row = df.iloc[i + 1, :]
            else:
                log("MISSING_END_ROW", i, None, "start(s) row not followed by an end(s) row", {"triplet_row": i})

            gloss_row = None
            if i - 1 >= 0 and _is_field(df.iat[i - 1, 1] if n_cols > 1 else None, "gloss_id :"):
                gloss_row = df.iloc[i - 1, :]

            # label is optional/unstable; do not error if missing
            label = None
            if gloss_row is not None and pd.notna(gloss_row.iat[0]):
                label = str(gloss_row.iat[0])
            elif pd.notna(starts_row.iat[0]):
                label = str(starts_row.iat[0])

            used_end_cols: set[int] = set()
            has_any_start = False

            for j in range(2, n_cols):
                sv = starts_row.iat[j]
                if pd.isna(sv):
                    continue

                has_any_start = True
                start = _safe_float(sv)
                if start is None:
                    log("NONNUMERIC_START", i, j, f"Start time is not numeric: {sv}", {"label": label})
                    continue

                gloss = None
                if gloss_row is not None:
                    gv = gloss_row.iat[j]
                    if pd.notna(gv):
                        gloss = str(gv)

                if ends_row is None:
                    log("NO_END_FOR_START", i, j, "No end row available to pair with this start",
                        {"label": label, "gloss": gloss, "start": start, "triplet_row": i})
                    continue

                end = None
                end_col = None
                for jj in range(j, n_cols):
                    if jj in used_end_cols:
                        continue
                    ev = ends_row.iat[jj]
                    if pd.notna(ev):
                        end_val = _safe_float(ev)
                        if end_val is None:
                            log("NONNUMERIC_END", i + 1, jj, f"End time is not numeric: {ev}",
                                {"label": label, "gloss": gloss, "start": start, "triplet_row": i})
                            end = None
                            end_col = None
                        else:
                            end = end_val
                            end_col = jj
                        break

                if end is None or end_col is None:
                    log("MISSING_END_VALUE", i + 1, j, "Could not find an end time to the right for this start",
                        {"label": label, "gloss": gloss, "start": start, "triplet_row": i})
                    continue

                used_end_cols.add(end_col)
                segments_seen += 1

                if end < start:
                    log("END_BEFORE_START", i, j, f"End ({end}) < Start ({start})",
                        {"label": label, "gloss": gloss, "start": start, "end": end, "triplet_row": i})

            # Empty start rows aren’t necessarily errors, but often indicate weird structure; keep as WARNING.
            if not has_any_start:
                log("EMPTY_START_ROW", i, None, "start(s) row contains no entries in columns >=2",
                    {"label": label, "triplet_row": i})

            i += 2 if ends_row is not None else 1
            continue

        i += 1

    if current_json is not None and not current_has_sentence:
        log("MISSING_SENTENCE", n_rows - 1, None, "File ended without a Korean sentence for last record")

    error_df = pd.DataFrame(errs)
    stats = {
        "source": source,
        "rows": n_rows,
        "cols": n_cols,
        "records_seen": records_seen,
        "records_with_sentence": records_with_sentence,
        "triplets_seen": triplets_seen,
        "segments_seen": segments_seen,
        "errors": len(errs),
    }
    return error_df, stats


def validate_xlsx_folder(folder: Path, pattern: str = "*.xlsx", limit: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_errs = []
    summaries = []

    files = sorted(folder.glob(pattern))
    if limit is not None:
        files = files[:limit]

    for f in files:
        try:
            df = pd.read_excel(f, header=None, engine="openpyxl")
        except Exception as e:
            all_errs.append(pd.DataFrame([{
                "source": str(f),
                "json": None,
                "row": None,
                "col": None,
                "code": "READ_FAIL",
                "message": repr(e),
            }]))
            summaries.append({
                "source": str(f),
                "rows": None,
                "cols": None,
                "records_seen": 0,
                "records_with_sentence": 0,
                "triplets_seen": 0,
                "segments_seen": 0,
                "errors": 1,
            })
            continue

        err_df, stats = validate_workbook_df(df, source=str(f))
        if not err_df.empty:
            all_errs.append(err_df)
        summaries.append(stats)

    all_errors_df = pd.concat(all_errs, ignore_index=True) if all_errs else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    return all_errors_df, summary_df


def sample_error_windows(
    errors_df: pd.DataFrame,
    max_samples: int = 10,
    rows_before: int = 1,
    rows_after: int = 2,
    cols_before: int = 2,
    cols_after_default: int = 3,
    min_col_start: int = 0,
    min_col_end: int = 8,
    code_cols_after: Optional[Dict[str, int]] = None,
) -> list[dict[str, Any]]:
    """
    Samples windows around each error and highlights the exact cell when possible.

    - Column window centres on the error 'col' (with before/after margin).
    - For certain error codes, automatically expands columns to the right.
    - Guarantees at least [min_col_start:min_col_end) if 'col' is missing.
    - Returns list of dicts with:
        source, json, code, message, row, col, row_range, col_range, window, window_hi
    """

    if code_cols_after is None:
        # Wider right-context for cases where the "next non-NaN to the right" matters
        code_cols_after = {
            "MISSING_END_VALUE": 15,
            "MISSING_END_ROW": 10,
            "END_BEFORE_START": 8,
            "NONNUMERIC_END": 10,
            "NONNUMERIC_START": 8,
        }

    samples: list[dict[str, Any]] = []
    if errors_df.empty:
        return samples

    # Prioritise actionable codes first (optional)
    priority = [
        "READ_FAIL",
        "MISSING_END_ROW",
        "MISSING_END_VALUE",
        "END_BEFORE_START",
        "NONNUMERIC_START",
        "NONNUMERIC_END",
        "MISSING_SENTENCE",
        "EMPTY_SENTENCE",
        "MISSING_JSON",
        "EMPTY_START_ROW",
    ]
    dfp = errors_df.copy()
    dfp["_prio"] = dfp["code"].apply(lambda x: priority.index(x) if x in priority else len(priority))
    dfp = dfp.sort_values(["_prio"]).drop(columns=["_prio"])

    for _, e in dfp.head(max_samples).iterrows():
        src = e.get("source")
        r = e.get("row")
        c = e.get("col")
        code = e.get("code")

        base = {k: e.get(k) for k in ["source", "json", "code", "message", "row", "col"]}

        # If row is missing (e.g., READ_FAIL), nothing to slice
        if pd.isna(r):
            samples.append(base)
            continue

        r = int(r)
        c_int = None if pd.isna(c) else int(c)

        # Load df fresh for this file
        df = pd.read_excel(src, header=None, engine="openpyxl")

        # Row window
        r0 = max(0, r - rows_before)
        r1 = min(len(df), r + rows_after + 1)

        # Code-aware column expansion to the right
        cols_after = code_cols_after.get(code, cols_after_default)

        # Column window
        if c_int is None:
            c0 = min_col_start
            c1 = min(df.shape[1], min_col_end)
        else:
            c0 = max(0, c_int - cols_before)
            c1 = min(df.shape[1], c_int + cols_after + 1)

            # Ensure a minimum visible width even if c_int is tiny
            min_width = (min_col_end - min_col_start)
            if c1 - c0 < min_width:
                c1 = min(df.shape[1], c0 + min_width)

        window = df.iloc[r0:r1, c0:c1]

        # Styled version highlighting the exact cell, if it’s inside the slice
        window_hi = None
        if c_int is not None and (r0 <= r < r1) and (c0 <= c_int < c1):
            rr = r - r0
            cc = c_int - c0

            def _hi(data: pd.DataFrame) -> pd.DataFrame:
                styles = pd.DataFrame("", index=data.index, columns=data.columns)
                styles.iat[rr, cc] = "background-color: yellow; font-weight: bold;"
                return styles

            window_hi = window.style.apply(_hi, axis=None)

        samples.append({
            **base,
            "row_range": (r0, r1),
            "col_range": (c0, c1),
            "window": window,
            "window_hi": window_hi,
        })

    return samples

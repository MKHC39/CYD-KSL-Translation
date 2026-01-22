from pathlib import Path
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
from pathmagic import smart_path as sp

def to_wsl_path(p: str) -> Path:
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

def inspect_array(arr: np.ndarray, name: str) -> None:
    print(f"\n== {name} ==")
    print("dtype:", arr.dtype)
    print("shape:", arr.shape)

    # Scalar (0-d) convenience
    if arr.ndim == 0:
        print("value:", arr.item())

    if np.issubdtype(arr.dtype, np.number):
        if arr.size:
            a = arr.astype(np.float64, copy=False)
            print("min/max:", a.min(), a.max())
            print("mean/std:", a.mean(), a.std())
            flat = arr.ravel()
            n = min(16, flat.size)
            print(f"preview[{n}]:", flat[:n])
    else:
        # For non-numeric arrays (strings/objects), just show a small preview
        flat = arr.ravel()
        n = min(16, flat.size)
        if flat.size:
            print(f"preview[{n}]:", flat[:n])

def inspect_npy_or_npz(path: Path) -> None:
    obj = safe_load(path, allow_pickle=False)
    if isinstance(obj, np.ndarray):
        #df = pd.DataFrame(obj)
        inspect_array(obj, path.name)
        print(type(obj.item()))
        #return df
    else:
        keys = list(obj.keys())
        print("archive keys:", keys)
        for k in keys:
            inspect_array(obj[k], k)

def safe_load(path: Path, allow_pickle: bool = False):
    try:
        return np.load(path, allow_pickle=allow_pickle)
    except ValueError as e:
        # Auto-retry for object arrays (common for gloss_dict.npy)
        if (not allow_pickle) and ("Object arrays cannot be loaded" in str(e)):
            print("[note] object array detected -> reloading with allow_pickle=True (trusted file only)")
            return np.load(path, allow_pickle=True)
        raise


# ---- quick interactive use ----
p = sp(r'\\wsl.localhost\Ubuntu\home\harrison\Workplace\workspaces\ksl\preprocessing\cache\NIA_SL_G1_COLDWAVE000070_1_TW07_R.npy')
if not p.exists():
    raise FileNotFoundError(p)
print("Resolved path:", p)
output = inspect_npy_or_npz(p)




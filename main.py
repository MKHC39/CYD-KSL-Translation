# placeholder



"""
ROOT = Path(r"C:\Users\CHOI\Downloads\KSL Word DataSet\수어 영상\1.Training\[라벨]01_real_word_keypoint\01")

IMG_W = 1920
IMG_H = 1080
MARGIN_PX = 20

N_WORDS = 100
ANGLES = ["D", "F", "L", "R", "U"]

# Folder name: NIA_SL_WORD0001_REAL01_D
FOLDER_RE = re.compile(r"^NIA_SL_WORD(\d{4})_REAL01_([DFLRU])$")

# File name: NIA_SL_WORD0001_REAL01_D_000000000000_keypoints.json
FILE_RE = re.compile(r"_([0-9]{12})_keypoints\.json$", re.IGNORECASE)


def has_any_xy_zero(keypoints_json: Path) -> bool:
    """
    Quick check: returns True if ANY (x==0 and y==0) occurs in face/left/right arrays.
    (This is a proxy for the common '0,0,0' missing-keypoint pattern.)
    """
    d = json.loads(keypoints_json.read_text(encoding="utf-8"))
    people = d["people"]

    for field in ("face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
        arr = people[field]
        n = len(arr)
        if n % 3 != 0:
            continue
        for i in range(0, n, 3):
            x = arr[i]
            y = arr[i + 1]
            if x == 0 and y == 0:
                return True
    return False

def list_word_folders(root: Path):
    """
    Returns dict: word_num(int) -> dict angle(str)->folder_path
    """
    word_map = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        m = FOLDER_RE.match(entry.name)
        if not m:
            continue
        word_num = int(m.group(1))
        angle = m.group(2)
        word_map.setdefault(word_num, {})[angle] = entry
    return word_map


def keypoint_files_sorted(folder: Path):
    """
    Sort by 12-digit frame index inside filename.
    """
    files = []
    for p in folder.glob("*_keypoints.json"):
        m = FILE_RE.search(p.name)
        if not m:
            continue
        frame_idx = int(m.group(1))
        files.append((frame_idx, p))
    files.sort(key=lambda t: t[0])
    return files

def main():
    word_map = list_word_folders(ROOT)
    if not word_map:
        raise RuntimeError(f"No matching word folders found under: {ROOT}")

    word_nums = sorted(word_map.keys())[:N_WORDS]

    for w in word_nums:
        print(f"\n=== WORD {w:04d} ===")
        angle_map = word_map[w]

        for angle in ANGLES:
            folder = angle_map.get(angle)
            if folder is None:
                print(f"[{angle}] MISSING folder")
                continue

            files = keypoint_files_sorted(folder)
            if not files:
                print(f"[{angle}] No keypoints files found")
                continue

            print(f"[{angle}] {folder.name}  files={len(files)}")

            # If you truly want every bbox printed, leave as-is.
            # Otherwise you can limit for sanity:
            # files = files[:10]

            zero_hits = 0
            for frame_idx, kp_path in files:
                bbox = box_outline(kp_path, IMG_W, IMG_H, margin_px=MARGIN_PX)
                if has_any_xy_zero(kp_path):
                    zero_hits += 1
                print(f"{folder.name} frame={frame_idx:012d} bbox={bbox}")

            print(f"[{angle}] frames_with_any_xy_zero={zero_hits}/{len(files)}")

"""

if __name__ == "__main__":
    main()

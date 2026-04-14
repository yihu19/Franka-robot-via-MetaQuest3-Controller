import os
import pickle
import cv2
import struct
import numpy as np
from typing import Tuple, List, Optional

# Unified dataset path (appending multiple episodes in one file)
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.abspath("./data"))
DATASET_PKL = os.path.join(DATASET_DIR, "dataset.pkl")
NEXT_ID_PATH = os.path.join(DATASET_DIR, "next_id.txt")
INSTR_CSV = os.path.join(DATASET_DIR, "instructions.csv")


# State layout (must match franka_teleop_controller._state_provider):
#   [0:7]   joint1..joint7 positions  (rad)
#   [7]     gripper trigger value      (0-1)
#   [8:15]  ee_x, ee_y, ee_z, ee_qx, ee_qy, ee_qz, ee_qw
#   [15:22] joint1_vel..joint7_vel     (rad/s)
EXPECTED_STATE_LEN = 22
SYNC_TOL_SEC = 0.1

# New: Index file utility (records the byte offset of each episode in pkl)
def _idx_path_for(pkl_path: str) -> str:
    return f"{pkl_path}.idx"

def _read_idx(idx_path: str, data_path: str) -> Optional[List[int]]:
    if not os.path.exists(idx_path) or not os.path.exists(data_path):
        return None
    try:
        st = os.stat(data_path)
        with open(idx_path, "rb") as f:
            hdr = f.read(24)  # size(8) + mtime_ns(8) + count(8)
            if len(hdr) != 24:
                return None
            size, mtime_ns, count = struct.unpack("<QQQ", hdr)
            if size != st.st_size or mtime_ns != getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)):
                return None
            buf = f.read(8 * count)
            if len(buf) != 8 * count:
                return None
            offsets = list(struct.unpack(f"<{count}Q", buf))
            return offsets
    except Exception:
        return None

def _write_idx(idx_path: str, data_path: str, offsets: List[int]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(idx_path)), exist_ok=True)
    st = os.stat(data_path)
    size = st.st_size
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    with open(idx_path, "wb") as f:
        f.write(struct.pack("<QQQ", size, mtime_ns, len(offsets)))
        if offsets:
            f.write(struct.pack(f"<{len(offsets)}Q", *offsets))

def _build_idx_by_scan(data_path: str) -> List[int]:
    """
    Sequentially scan pkl and record the offset at the start of each object.
    Note: Need to pickle.load one by one, the first time will be slower, but only needs to be done once.
    """
    offsets: List[int] = []
    with open(data_path, "rb") as f:
        i = 0
        while True:
            pos = f.tell()
            try:
                obj = pickle.load(f)
                offsets.append(pos)
                i += 1
                if i % 50 == 0:
                    print(f"[pkl-index] indexed {i} episodes...")
            except EOFError:
                break
    print(f"[pkl-index] total episodes indexed: {len(offsets)}")
    return offsets

def _ensure_idx(data_path: str) -> Optional[List[int]]:
    idx_path = _idx_path_for(data_path)
    offsets = _read_idx(idx_path, data_path)
    if offsets is not None:
        return offsets
    # Build and save index
    try:
        print(f"[pkl-index] building index for {data_path} ...")
        offsets = _build_idx_by_scan(data_path)
        _write_idx(idx_path, data_path, offsets)
        print(f"[pkl-index] index saved -> {idx_path}")
        return offsets
    except Exception as e:
        print(f"[pkl-index] failed to build index: {e}")
        return None


def iter_episodes(path):
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        i = 0
        while True:
            try:
                obj = pickle.load(f)
                yield obj
                i += 1
            except EOFError:
                break
            except Exception as e:
                raise RuntimeError(f"Unpickle failed at object #{i} in {path}: {e}")

def _count_episodes_in_pickle(path: str) -> int:
    # Prefer using index
    offsets = _ensure_idx(path)
    if offsets is not None:
        return len(offsets)
    # Fallback: Sequential scan count (memory-friendly, but requires unpacking once)
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, "rb") as f:
        while True:
            try:
                pickle.load(f)
                cnt += 1
            except EOFError:
                break
            except Exception as e:
                raise RuntimeError(f"Counting pickle objects failed at #{cnt} in {path}: {e}")
    return cnt

# Used to generate the next ID: Prefer reading from the pointer file; otherwise count the number of existing episodes in dataset.pkl
def get_next_episode_id(save_dir: str = None) -> int:
    os.makedirs(DATASET_DIR, exist_ok=True)
    if os.path.exists(NEXT_ID_PATH):
        try:
            return int(open(NEXT_ID_PATH, "r").read().strip())
        except Exception:
            pass
    return _count_episodes_in_pickle(DATASET_PKL)

def load_episode(path: str, index: int = -1):
    """
    Supports loading a single entry from large files:
    - If an .idx index file exists (or is successfully built for the first time), seek to the target offset to read one episode.
    - Otherwise, fallback to stream reading until the target index (will not load all episodes into memory).
    Returns: (episode_obj, steps_list)
    """
    # Prefer using index for positioning
    offsets = _ensure_idx(path)
    if offsets is not None and len(offsets) > 0:
        N = len(offsets)
        # Normalize index
        if index is None:
            index = -1
        if index < 0:
            index = N + index
        if index < 0 or index >= N:
            raise IndexError(f"episode index out of range: {index} (valid: 0..{N-1}) in {path}")
        with open(path, "rb") as f:
            f.seek(offsets[index])
            ep = pickle.load(f)
    else:
        # No index: Stream read until target index
        if not os.path.exists(path):
            raise RuntimeError(f"No episode found in {path}")
        # Negative index needs to get N first (will scan once, but not cache)
        if index is None:
            index = -1
        if index < 0:
            N = _count_episodes_in_pickle(path)
            index = N + index
        if index < 0:
            raise IndexError(f"episode index out of range: {index} in {path}")
        ep = None
        cur = 0
        with open(path, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                except EOFError:
                    break
                if cur == index:
                    ep = obj
                    break
                cur += 1
        if ep is None:
            raise IndexError(f"episode index out of range: {index} in {path}")

    # Compatible with {"steps":[...]} or direct steps list
    if isinstance(ep, dict) and "steps" in ep:
        steps = ep["steps"]
    elif isinstance(ep, list):
        steps = ep
    else:
        raise RuntimeError(f"Unsupported episode type: {type(ep)}")
    return ep, steps

def decode_image(img):
    # Supports bytes (jpg/jpeg/png) or ndarray (BGR)
    if isinstance(img, bytes):
        arr = np.frombuffer(img, dtype=np.uint8)
        dec = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if dec is None:
            raise ValueError("Failed to decode image bytes")
        return dec
    if isinstance(img, np.ndarray):
        return img
    raise TypeError(f"Unsupported image type: {type(img)}")

def get_obs(step: dict):
    # Compatible key names
    obs = step.get("observation", step)
    images = obs.get("images", {})
    img1 = images.get("image_side", None)
    img2 = images.get("image_wrist", None)
    img3 = images.get("image_head", None)
    state = obs.get("qpos", None)
    return img1, img2, img3, state

def get_all_obs(steps: dict):
    # Compatible key names
    obs = steps.get("observation", steps)
    images = obs.get("images", {})
    img1 = images.get("image_side", None)
    img2 = images.get("image_wrist", None)
    img3 = images.get("image_head", None)
    qpos = obs.get("qpos", None)
    qvel = obs.get("qvel", None)
    # ee_pose = obs.get("ee_pose", None)
    return img1, img2, img3, qpos, qvel

def make_mosaic(img1, img2, img3, target_h=448) -> np.ndarray:
    if img1 is None and img2 is None and img3 is None:
        return None
    imgs = []
    for im in (img1, img2, img3):
        if im is None:
            continue
        h, w = im.shape[:2]
        scale = target_h / h
        imr = cv2.resize(im, (int(w * scale), target_h))
        imgs.append(imr)
    if not imgs:
        return None
    if len(imgs) == 1:
        return imgs[0]
    return np.hstack(imgs)

def render_trajectory(states: np.ndarray, size: Tuple[int, int] = (800, 280), max_dims: int = 6) -> np.ndarray:
    """
    Draw the entire state as a line chart (up to the first max_dims dimensions)
    """
    if states.ndim == 1:
        states = states[:, None]
    T, D = states.shape
    Dp = min(D, max_dims)
    W, H = size
    pad = 30
    img = np.full((H, W, 3), 20, np.uint8)

    # Normalize to [0,1] (for each dimension separately)
    s_min = states.min(axis=0, keepdims=True)
    s_max = states.max(axis=0, keepdims=True)
    span = np.maximum(s_max - s_min, 1e-6)
    norm = (states - s_min) / span

    xs = np.linspace(pad, W - pad, T).astype(int)
    colors = [(255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0), (186, 85, 211), (0, 206, 209)]

    # Axes
    cv2.rectangle(img, (pad, pad), (W - pad, H - pad), (80, 80, 80), 1)
    cv2.putText(img, "State Trajectory (first {} dims)".format(Dp),
                (pad, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    for d in range(Dp):
        ys = (H - pad - (H - 2 * pad) * norm[:, d]).astype(int)
        pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, colors[d % len(colors)], 2, cv2.LINE_AA)
        label = f"d{d} [{s_min[0,d]:.2f},{s_max[0,d]:.2f}]"
        cv2.putText(img, label, (pad + d * 120, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[d % len(colors)], 1, cv2.LINE_AA)

    return img

def open_writer(path: str, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter | None, str]:
    # Ensure the directory exists
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    # Prefer mp4, fallback to avi
    for ext, fourcc_str in (("mp4", "mp4v"), ("mp4", "avc1"), ("avi", "XVID")):
        out_path = path
        if not path.lower().endswith(f".{ext}"):
            if ext == "avi":
                out_path = os.path.splitext(path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(out_path, fourcc, max(fps, 1.0), size)
        if vw.isOpened():
            return vw, out_path
    return None, path
    
def fmt(arr, max_items=7):
    parts = []
    for v in arr[:max_items]:
        try:
            parts.append(f"{float(v):.2f}")
        except Exception:
            parts.append(str(v))
    return ", ".join(parts)

def format_state_preview(state, max_items=7):
    """Preview text to overlay on the image: values retained to two decimal places; non-numeric converted to string."""
    arr = np.asarray(state).ravel()
    return fmt(arr, max_items), arr.size

# New: Format q/qdot and determine gripper status based on the last dimension of dq
def format_pos_vel_preview(qpos, qvel, max_items=7, zero_eps: float = 1e-3):
    qp = np.asarray(qpos).ravel() if qpos is not None else np.array([])
    qv = np.asarray(qvel).ravel() if qvel is not None else np.array([])

    # Gripper status: take the last dimension of dq
    gripper = "unknown"
    if qv.size > 0:
        last = float(qv[-1])
        if last > zero_eps:
            gripper = "opening"
        elif last < -zero_eps:
            gripper = "closing"
        elif qp[-1] < 1.0 and abs(last) <= zero_eps:
            gripper = "grasped"
        else:
            gripper = "free"

    return fmt(qp, max_items), fmt(qv, max_items), qp.size, qv.size, gripper

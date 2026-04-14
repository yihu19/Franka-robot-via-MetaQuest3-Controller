import os
import csv
import pickle
import argparse
import fcntl  # Linux file lock to prevent concurrent write conflicts
from typing import Iterable, List, Optional, Set, Tuple

from load_data_utils import (
    _idx_path_for,
    _write_idx,
    _ensure_idx,
    DATASET_PKL, 
    )

# -------- Low overhead tools --------
def _episode_count_from_idx(pkl_path: str) -> int:
    return len(_ensure_idx(pkl_path))

def _load_obj_at_offset(pkl_path: str, offset: int):
    """Deserialize a single object at the given offset."""
    with open(pkl_path, "rb") as f:
        f.seek(offset)
        up = pickle.Unpickler(f)
        return up.load()

def _get_episode_id_from_obj(obj) -> Optional[int]:
    if isinstance(obj, dict):
        meta = obj.get("metadata") or {}
        eid = meta.get("episode_id")
        try:
            return int(eid) if eid is not None else None
        except Exception:
            return None
    return None

def _collect_episode_ids_by_indices(pkl_path: str, offsets: List[int], indices: Iterable[int]) -> Set[int]:
    """Get the set of episode_ids to be deleted for CSV cleanup; skip and warn if corrupted."""
    ids: Set[int] = set()
    for i in sorted(set(indices)):
        try:
            obj = _load_obj_at_offset(pkl_path, offsets[i])
            eid = _get_episode_id_from_obj(obj)
            if eid is not None:
                ids.add(eid)
        except Exception as e:
            print(f"[delete] warn: cannot read episode_id at index {i}: {e}")
    return ids

# -------- Core: Copy and keep byte ranges by idx, atomically replace and rewrite idx --------
def _copy_keep_ranges_with_idx(pkl_path: str, keep_indices: List[int]) -> Tuple[int, int]:
    """
    Copy only the retained entries based on idx (by byte range), returning (kept_cnt, removed_cnt).
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)
    offsets = _ensure_idx(pkl_path)
    N = len(offsets)
    keep_set = sorted(set(i for i in keep_indices if 0 <= i < N))
    removed_cnt = N - len(keep_set)

    tmp_path = pkl_path + ".tmp"
    new_offsets: List[int] = []
    cur_off = 0
    copied = 0

    with open(pkl_path, "rb") as fin, open(tmp_path, "wb") as fout:
        # Exclusive lock to avoid concurrent writes
        fcntl.flock(fin.fileno(), fcntl.LOCK_EX)
        fcntl.flock(fout.fileno(), fcntl.LOCK_EX)

        CHUNK = 8 * 1024 * 1024
        for idx in keep_set:
            start = offsets[idx]
            end = offsets[idx + 1] if idx + 1 < N else os.path.getsize(pkl_path)
            if end <= start:
                continue
            size = end - start
            # Record the starting offset in the new file
            new_offsets.append(cur_off)
            # Copy the original byte range of this object
            fin.seek(start)
            remain = size
            while remain > 0:
                buf = fin.read(min(CHUNK, remain))
                if not buf:
                    break
                fout.write(buf)
                remain -= len(buf)
            cur_off += size
            copied += 1

        fout.flush(); os.fsync(fout.fileno())
        fcntl.flock(fout.fileno(), fcntl.LOCK_UN)
        fcntl.flock(fin.fileno(), fcntl.LOCK_UN)

    # Atomic replacement
    os.replace(tmp_path, pkl_path)

    # Rewrite idx (matching new file stat)
    idx_path = _idx_path_for(pkl_path)
    _write_idx(idx_path, pkl_path, new_offsets)
    kept_cnt = copied
    return kept_cnt, removed_cnt

# -------- CSV update --------
def _filter_csv_by_episode_ids(csv_path: str, ids_to_remove: Set[int]) -> int:
    if not csv_path or not os.path.exists(csv_path) or not ids_to_remove:
        return 0
    tmp = csv_path + ".tmp"
    removed = 0
    # Handle BOM
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fin, \
         open(tmp, "w", encoding="utf-8", newline="") as fout:
        fcntl.flock(fout.fileno(), fcntl.LOCK_EX)
        r = csv.reader(fin); w = csv.writer(fout)
        rows = list(r)
        if rows:
            w.writerow(rows[0])
            for row in rows[1:]:
                if not row:
                    continue
                try:
                    eid = int(row[0])
                except Exception:
                    eid = None
                if eid is not None and eid in ids_to_remove:
                    removed += 1
                    continue
                w.writerow(row)
        fout.flush(); os.fsync(fout.fileno())
        fcntl.flock(fout.fileno(), fcntl.LOCK_UN)
    os.replace(tmp, csv_path)
    print(f"[delete] CSV updated -> {csv_path} (removed {removed} rows)")
    return removed

# -------- next_id.txt update (without deserializing pkl)--------
def _max_episode_id_from_csv(csv_path: Optional[str]) -> Optional[int]:
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        mx = None
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            rows = list(r)
            if not rows:
                return None
            for row in rows[1:]:
                if not row:
                    continue
                try:
                    eid = int(row[0])
                    mx = eid if mx is None else max(mx, eid)
                except Exception:
                    continue
        return mx
    except Exception:
        return None

def _write_next_id(next_id_path: str, value: int):
    os.makedirs(os.path.dirname(os.path.abspath(next_id_path)), exist_ok=True)
    with open(next_id_path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(str(int(value)).strip() + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    print(f"[delete] next_id.txt updated -> {next_id_path} (value={int(value)})")

def _update_next_id_file_by_idx_or_csv(pkl_path: str, next_id_path: str, csv_path: Optional[str]) -> int:
    mx = _max_episode_id_from_csv(csv_path)
    if mx is not None:
        next_id = int(mx) + 1
        _write_next_id(next_id_path, next_id)
        return next_id
    N = len(_ensure_idx(pkl_path))
    next_id = N
    _write_next_id(next_id_path, next_id)
    return next_id

# -------- External API (including next_id update)--------
def delete_by_indices(pkl_path: str, indices: Iterable[int], instr_csv_path: Optional[str] = None,
                      next_id_path: Optional[str] = None) -> Tuple[int, int, int, Optional[int]]:
    """
    Delete by index: returns (removed_cnt, kept_cnt, csv_removed_cnt, new_next_id).
    """
    offsets = _ensure_idx(pkl_path)
    N = len(offsets)
    to_del = sorted(set(i for i in indices if 0 <= i < N))
    if not to_del:
        new_next = _update_next_id_file_by_idx_or_csv(pkl_path, next_id_path, instr_csv_path) if next_id_path else None
        return 0, N, 0, new_next

    ids_to_remove = _collect_episode_ids_by_indices(pkl_path, offsets, to_del)

    keep = [i for i in range(N) if i not in to_del]
    kept_cnt, removed_cnt = _copy_keep_ranges_with_idx(pkl_path, keep)

    csv_removed = _filter_csv_by_episode_ids(instr_csv_path, ids_to_remove) if instr_csv_path else 0
    new_next = _update_next_id_file_by_idx_or_csv(pkl_path, next_id_path, instr_csv_path) if next_id_path else None
    return removed_cnt, kept_cnt, csv_removed, new_next

def delete_by_episode_ids(pkl_path: str, episode_ids: Iterable[int], instr_csv_path: Optional[str] = None,
                          next_id_path: Optional[str] = None) -> Tuple[int, int, int, Optional[int]]:
    """
    Delete by episode_id: returns (removed_cnt, kept_cnt, csv_removed_cnt, new_next_id).
    """
    ids = set(int(e) for e in episode_ids)
    offsets = _ensure_idx(pkl_path)
    N = len(offsets)

    to_del: List[int] = []
    with open(pkl_path, "rb") as f:
        up = pickle.Unpickler(f)
        for i in range(N):
            f.seek(offsets[i])
            try:
                obj = up.load()
            except Exception as e:
                print(f"[delete] warn: skip unreadable object at index {i}: {e}")
                continue
            eid = _get_episode_id_from_obj(obj)
            if eid is not None and eid in ids:
                to_del.append(i)

    if not to_del:
        new_next = _update_next_id_file_by_idx_or_csv(pkl_path, next_id_path, instr_csv_path) if next_id_path else None
        return 0, N, 0, new_next

    return delete_by_indices(pkl_path, to_del, instr_csv_path, next_id_path)

def main():
    ap = argparse.ArgumentParser(description="Delete episode(s) using .idx offsets (no full unpickle)")
    ap.add_argument("--pkl", default=DATASET_PKL, help="dataset pickle path")
    ap.add_argument("--csv", help="instruction csv path (default: <pkl_dir>/instructions.csv)")
    ap.add_argument("--next-id", help="next_id.txt path (default: <pkl_dir>/next_id.txt)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, nargs="+", help="0-based index(es) to delete")
    g.add_argument("--episode", type=int, nargs="+", dest="eid", help="episode_id(s) to delete")
    ap.add_argument("--dry-run", action="store_true", help="show what would be deleted without writing")
    args = ap.parse_args()

    pkl_path = args.pkl
    base_dir = os.path.dirname(os.path.abspath(pkl_path))
    csv_path = args.csv if args.csv else os.path.join(base_dir, "instructions.csv")
    next_id_path = args.next_id if args.next_id else os.path.join(base_dir, "next_id.txt")

    if args.index:
        if args.dry_run:
            print(f"[dry-run] would delete indices: {sorted(set(args.index))}")
            return
        removed, kept, csv_removed, new_next = delete_by_indices(pkl_path, args.index, csv_path, next_id_path)
    else:
        if args.dry_run:
            print(f"[dry-run] would delete episode_ids: {sorted(set(args.eid))}")
            return
        removed, kept, csv_removed, new_next = delete_by_episode_ids(pkl_path, args.eid, csv_path, next_id_path)

    msg_next = f", next_id={new_next}" if new_next is not None else ""
    print(f"[delete] removed={removed}, kept={kept}, csv_removed={csv_removed}{msg_next}")
    
if __name__ == "__main__":
    main()
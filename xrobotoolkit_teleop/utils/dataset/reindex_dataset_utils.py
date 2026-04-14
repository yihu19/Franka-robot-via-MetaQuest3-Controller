import os
import csv
import pickle
import argparse
import fcntl
import gc
import time
from typing import Optional
try:
    import psutil
except ImportError:
    psutil = None

import io
import pickletools

from load_data_utils import DATASET_PKL, INSTR_CSV, NEXT_ID_PATH, _ensure_idx

def _rss():
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss

def _format_bytes(n):
    if n is None: return "NA"
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def _update_next_id(next_id_path: str, next_id: int) -> None:
    if not next_id_path:
        return
    os.makedirs(os.path.dirname(next_id_path) or ".", exist_ok=True)
    tmp = next_id_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(int(next_id)).strip() + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, next_id_path)
    print(f"[next_id] -> {next_id}")

def reindex_idx_only(pkl_path: str,
                     csv_path: Optional[str],
                     next_id_path: Optional[str],
                     start_id: int,
                     map_out: Optional[str]) -> None:
    """
    Sequential re-indexing only (does not modify PKL):
      - Generate new continuous episode_id based on the number of idx
      - Stream read CSV and rewrite the first column (episode_id)
      - Write out next_id
      - Write out mapping file old_id,new_id (optional)
    """
    offsets = _ensure_idx(pkl_path)
    N = len(offsets)
    print(f"[idx-only] episodes={N}")

    id_map_path = map_out or os.path.join(os.path.dirname(pkl_path), "id_map.csv")
    tmp_csv = None
    if csv_path and os.path.exists(csv_path):
        tmp_csv = csv_path + ".tmp"
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as fin, \
             open(tmp_csv, "w", encoding="utf-8", newline="") as fout, \
             open(id_map_path, "w", encoding="utf-8", newline="") as fmap:
            r = csv.reader(fin); w = csv.writer(fout); m = csv.writer(fmap)
            rows_iter = iter(r)
            header = next(rows_iter, None)
            if header:
                w.writerow(header)
                m.writerow(["old_episode_id","new_episode_id"])
            seq = 0
            for row in rows_iter:
                if not row:
                    continue
                try:
                    old = int(row[0])
                except Exception:
                    continue
                if seq >= N:
                    # Discard rows in CSV that exceed those in PKL
                    continue
                new_id = start_id + seq
                row[0] = str(new_id)
                w.writerow(row)
                m.writerow([old, new_id])
                seq += 1
        os.replace(tmp_csv, csv_path)
        print(f"[idx-only] CSV updated -> {csv_path}, mapped={seq}")
        print(f"[idx-only] id_map -> {id_map_path}")
    else:
        # Still generate a pure sequential mapping (no old id) for future reference
        with open(id_map_path, "w", encoding="utf-8", newline="") as fmap:
            m = csv.writer(fmap)
            m.writerow(["old_episode_id","new_episode_id"])
            for i in range(N):
                m.writerow([None, start_id + i])
        print(f"[idx-only] id_map(no-old) -> {id_map_path}")

    _update_next_id(next_id_path, start_id + N)

def _patch_episode_id(raw: bytes, new_id: int) -> bytes:
    """
    Perform byte-level replacement on raw pickle only when the pattern is met:
    - Find SHORT_BINUNICODE 'episode_id'
    - Find the subsequent integer opcode (BININT1/BININT2/BININT/BINLONG)
    - If the new integer can be encoded with the same or smaller size, replace and return new raw
    - Otherwise return None to let the upper layer fall back to full deserialization
    """
    key = b"episode_id"
    pos = raw.find(key)
    if pos == -1:
        return raw  # No such key, no need to modify
    # Forward locate opcode SHORT_BINUNICODE (0x8c)
    # pickle format: 0x8c <len=1byte> <utf8 bytes> (len=10)
    # After finding the key, scan the opcode sequence to get the first integer
    # Use pickletools to parse and record the starting offset of the first int after the episode_id key
    try:
        stream = io.BytesIO(raw)
        target_offset = None
        after_key = False
        for opcode, arg, o_pos in pickletools.genops(stream):
            if opcode.name == "SHORT_BINUNICODE" and arg == "episode_id":
                after_key = True
                continue
            if after_key:
                # The integer could be BININT1/BININT2/BININT/LONG1/LONG4
                if opcode.name in ("BININT1","BININT2","BININT","LONG1","LONG4"):
                    target_offset = o_pos
                    break
                # Encountering other key values indicates a different structure, give up
                if opcode.name == "SHORT_BINUNICODE":
                    break
        if target_offset is None:
            return raw  # No replaceable integer found
        # Parse the original integer encoding
        b = bytearray(raw)
        op = b[target_offset]
        if op == 0x4f:  # BININT1
            if 0 <= new_id < 256:
                b[target_offset+1] = new_id
                return bytes(b)
        elif op == ord('M'):  # BININT2
            if 0 <= new_id < 65536:
                b[target_offset+1] = new_id & 0xff
                b[target_offset+2] = (new_id >> 8) & 0xff
                return bytes(b)
        elif op == ord('J'):  # BININT (4 bytes little-endian signed)
            if -2**31 <= new_id < 2**31:
                for i in range(4):
                    b[target_offset+1+i] = (new_id >> (8*i)) & 0xff
                return bytes(b)
        elif op in (0x8a,0x8b):  # LONG1 / LONG4 variable length is complex, give up directly
            return None
    except Exception:
        return None
    return None

def reindex_patch(pkl_path: str,
                  csv_path: Optional[str],
                  next_id_path: Optional[str],
                  start_id: int,
                  map_out: Optional[str],
                  fallback_load: bool = True,
                  report_interval: int = 50) -> None:
    """
    Patch mode: modify episode_id at the byte level without deserializing large data.
    Fallback: if safe replacement fails and fallback_load=True, load+dump one at a time.
    """
    offsets = _ensure_idx(pkl_path)
    N = len(offsets)
    print(f"[patch] episodes={N}")
    id_map_path = map_out or os.path.join(os.path.dirname(pkl_path), "id_map.csv")
    tmp_pkl = pkl_path + ".patch.tmp"
    with open(pkl_path, "rb") as fin, \
         open(tmp_pkl, "wb") as fout, \
         open(id_map_path, "w", encoding="utf-8", newline="") as fmap:
        fcntl.flock(fin.fileno(), fcntl.LOCK_EX)
        fcntl.flock(fout.fileno(), fcntl.LOCK_EX)
        m = csv.writer(fmap); m.writerow(["old_episode_id","new_episode_id"])
        written = 0
        for i, off in enumerate(offsets):
            start = off
            end = offsets[i+1] if i+1 < N else os.path.getsize(pkl_path)
            size = end - start
            fin.seek(start)
            raw = fin.read(size)
            old_id = None
            # Attempt to parse old id (lightweight): regex search for the integer ASCII immediately following episode_id (suitable for protocol < 3 text format I<int>\n)
            # If it fails, keep None
            # Here, obtaining old_id is not mandatory, only used for mapping file
            patched = _patch_episode_id(raw, start_id + written)
            if patched is None and fallback_load:
                # Safe fallback: only load one
                try:
                    ep = pickle.loads(raw)
                    if isinstance(ep, dict):
                        meta = ep.get("metadata") or {}
                        old_id = meta.get("episode_id")
                        meta["episode_id"] = start_id + written
                        ep["metadata"] = meta
                    patched = pickle.dumps(ep, protocol=pickle.HIGHEST_PROTOCOL)
                    del ep
                except Exception as e:
                    print(f"[patch] fallback failed idx={i}: {e}")
                    patched = raw  # Keep original content, old id unchanged
            if patched is not None:
                fout.write(patched)
            else:
                fout.write(raw)
            m.writerow([old_id, start_id + written])
            written += 1
            if written % report_interval == 0:
                rss = _rss()
                print(f"[patch] progress {written}/{N} rss={_format_bytes(rss)}")
        fout.flush(); os.fsync(fout.fileno())
        fcntl.flock(fout.fileno(), fcntl.LOCK_UN)
        fcntl.flock(fin.fileno(), fcntl.LOCK_UN)
    os.replace(tmp_pkl, pkl_path)
    print(f"[patch] PKL rewritten -> {pkl_path}")
    # CSV only modifies the first column to the new id (assuming order aligns with PKL)
    if csv_path and os.path.exists(csv_path):
        tmp_csv = csv_path + ".tmp"
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as fin, \
             open(tmp_csv, "w", encoding="utf-8", newline="") as fout:
            r = csv.reader(fin); w = csv.writer(fout)
            header = next(r, None)
            if header: w.writerow(header)
            new_id = start_id
            for row in r:
                if not row: continue
                row[0] = str(new_id)
                w.writerow(row)
                new_id += 1
        os.replace(tmp_csv, csv_path)
        print(f"[patch] CSV updated -> {csv_path}")
    if next_id_path:
        _update_next_id(next_id_path, start_id + written)
    print(f"[patch] id_map -> {id_map_path}")

def reindex_deep(pkl_path: str,
                 csv_path: Optional[str],
                 next_id_path: Optional[str],
                 start_id: int,
                 map_out: Optional[str],
                 gc_interval: int = 50) -> None:
    """
    Deep rewrite: truly modify each object's metadata.episode_id.
    Memory optimization:
      - Process one at a time, immediately dump
      - Do not keep the complete id_map in memory, write directly to disk
      - Periodically gc.collect()
      - Print RSS
    """
    offsets = _ensure_idx(pkl_path)
    N = len(offsets)
    print(f"[deep] episodes={N}")

    tmp_pkl = pkl_path + ".reindex.tmp"
    id_map_path = map_out or os.path.join(os.path.dirname(pkl_path), "id_map.csv")

    # Open three files: source read, target write, mapping write
    with open(pkl_path, "rb") as fin, \
         open(tmp_pkl, "wb") as fout, \
         open(id_map_path, "w", encoding="utf-8", newline="") as fmap:

        fcntl.flock(fin.fileno(), fcntl.LOCK_EX)
        fcntl.flock(fout.fileno(), fcntl.LOCK_EX)

        m = csv.writer(fmap)
        m.writerow(["old_episode_id","new_episode_id"])

        up = pickle.Unpickler(fin)
        written = 0
        last_gc_t = time.time()

        for seq, off in enumerate(offsets):
            fin.seek(off)
            try:
                ep = up.load()
            except Exception as e:
                print(f"[deep] skip idx={seq} off={off}: {e}")
                continue

            old_id = None
            if isinstance(ep, dict):
                meta = ep.get("metadata") or {}
                old_id = meta.get("episode_id")
                meta["episode_id"] = start_id + written
                ep["metadata"] = meta
            else:
                # If the structure is incompatible, just continue (do not write to mapping)
                pass

            pickle.dump(ep, fout, protocol=pickle.HIGHEST_PROTOCOL)
            m.writerow([old_id, start_id + written])
            written += 1

            # Release current object memory
            del ep
            # Periodic GC
            if written % gc_interval == 0:
                gc.collect()
                rss = _rss()
                print(f"[deep] progress {written}/{N} rss={_format_bytes(rss)}")

        fout.flush(); os.fsync(fout.fileno())
        fcntl.flock(fout.fileno(), fcntl.LOCK_UN)
        fcntl.flock(fin.fileno(), fcntl.LOCK_UN)

    # Atomic replacement
    os.replace(tmp_pkl, pkl_path)
    print(f"[deep] PKL rewritten -> {pkl_path}, written={written}")

    # Rewrite CSV (streaming, no id_map in memory)
    if csv_path and os.path.exists(csv_path):
        tmp_csv = csv_path + ".tmp"
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as fin, \
             open(tmp_csv, "w", encoding="utf-8", newline="") as fout, \
             open(id_map_path, "r", encoding="utf-8", newline="") as fmap:
            # Build an iterator for old->new mapping (stream read id_map file)
            map_reader = csv.reader(fmap)
            next(map_reader, None)  # skip header
            id_iter = ( (int(r[0]) if r[0] not in ("", "None") else None,
                         int(r[1])) for r in map_reader )
            # Build a lazy dictionary batch cache for old_id lookup (avoid complete read)
            # Simplified: since the CSV row order usually aligns with the write order, consume directly
            r_csv = csv.reader(fin)
            w_csv = csv.writer(fout)
            header = next(r_csv, None)
            if header:
                w_csv.writerow(header)
            # Read mapping rows
            map_current = next(id_iter, None)
            for row in r_csv:
                if not row:
                    continue
                try:
                    old = int(row[0])
                except Exception:
                    continue
                # Sequential matching: if the current mapping row matches old_id, replace
                while map_current and map_current[0] is not None and map_current[0] < old:
                    map_current = next(id_iter, None)
                if map_current and map_current[0] == old:
                    row[0] = str(map_current[1])
                    map_current = next(id_iter, None)
                    w_csv.writerow(row)
                # If old is None in the mapping (previously damaged skip), the row is discarded
        os.replace(tmp_csv, csv_path)
        print(f"[deep] CSV updated -> {csv_path}")

    if next_id_path:
        _update_next_id(next_id_path, start_id + written)
    print(f"[deep] id_map -> {id_map_path}")

# main adds patch mode
def main():
    ap = argparse.ArgumentParser(description="Memory-efficient reindex (PKL + CSV)")
    ap.add_argument("--pkl", default=DATASET_PKL)
    ap.add_argument("--csv", default=INSTR_CSV)
    ap.add_argument("--next-id", default=NEXT_ID_PATH)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--mode", choices=["idx-only","deep","map-only","patch"], default="patch",
                    help="patch: byte-level modification of metadata, does not deserialize large arrays")
    ap.add_argument("--map-out", help="Mapping file path (default <pkl_dir>/id_map.csv)")
    ap.add_argument("--gc-interval", type=int, default=50)
    ap.add_argument("--no-fallback", action="store_true", help="Do not fallback load+dump on patch failure")
    args = ap.parse_args()
    base_dir = os.path.dirname(os.path.abspath(args.pkl))
    map_out = args.map_out or os.path.join(base_dir, "id_map.csv")
    if args.mode == "patch":
        reindex_patch(args.pkl, args.csv, args.next_id, args.start_id, map_out,
                      fallback_load=not args.no_fallback)
        return
    if args.mode == "map-only":
        offsets = _ensure_idx(args.pkl)
        with open(map_out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["old_episode_id","new_episode_id"])
            for i in range(len(offsets)):
                w.writerow([None, args.start_id + i])
        print(f"[map-only] id_map -> {map_out}, episodes={len(offsets)}")
        _update_next_id(args.next_id, args.start_id + len(offsets))
        return
    if args.mode == "idx-only":
        reindex_idx_only(args.pkl, args.csv, args.next_id, args.start_id, map_out)
    else:
        reindex_deep(args.pkl, args.csv, args.next_id, args.start_id, map_out, gc_interval=args.gc_interval)

if __name__ == "__main__":
    main()
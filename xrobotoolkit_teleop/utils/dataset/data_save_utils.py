import os
import csv
import cv2
import fcntl  # Linux file lock to prevent concurrent write conflicts
import pickle
import numpy as np

def center_crop_and_resize(image, crop_h, crop_w, out_h=448, out_w=448):
    h, w = image.shape[:2]
    start_y = max((h - crop_h) // 2, 0)
    start_x = max((w - crop_w) // 2, 0)
    cropped = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    resized = cv2.resize(cropped, (out_w, out_h))
    return resized

def is_numeric_state(x, expected_len: int = None) -> bool:
    try:
        arr = np.asarray(x, dtype=np.float64).ravel()
        if expected_len is not None and arr.size != expected_len:
            return False
        return arr.size > 0 and np.all(np.isfinite(arr))
    except Exception:
        return False

def append_pickle(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "ab") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def append_instruction_csv(path: str, episode_id: int, instruction: str, num_steps: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    need_header = not os.path.exists(path)
    with open(path, "a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["episode_id", "language_instruction", "num_steps", ])
        writer.writerow([episode_id, instruction, num_steps])
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# Raw -> BGR array (H,W,3), no cropping or resizing
def raw_to_bgr(raw):
    if raw is None:
        return None
    data_bytes, w, h, enc = raw
    enc = (enc or '').lower()
    try:
        if enc in ('yuv422_yuy2', 'yuyv'):
            yuyv = np.frombuffer(data_bytes, np.uint8).reshape(h, w, 2)
            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        elif enc == 'bgr8':
            bgr = np.frombuffer(data_bytes, np.uint8).reshape(h, w, 3)
        else:
            # Fallback: convert to BGR8 via cv_bridge (memoryview -> bytes)
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image as ImgMsg
            bridge = CvBridge()
            msg = ImgMsg()
            msg.height = h
            msg.width = w
            msg.encoding = enc
            msg.is_bigendian = 0
            msg.step = int(len(data_bytes) // h)
            msg.data = bytes(data_bytes)
            bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Ensure contiguous & uint8
        if bgr.dtype != np.uint8:
            bgr = bgr.astype(np.uint8, copy=False)
        return np.ascontiguousarray(bgr)
    except Exception:
        return None
    
# Directly encode raw (prefer YUYV) to JPEG bytes; fallback to BGR->JPEG on failure
def raw_to_jpg(raw, quality: int = 90):
    if raw is None:
        return None
    data_bytes, w, h, enc = raw
    enc = (enc or '').lower()

    # # Branch 1: YUYV via libjpeg-turbo (YUV422 planar), avoid intermediate BGR conversion
    # if enc in ('yuv422_yuy2', 'yuyv'):
    #     try:
    #         from turbojpeg import TurboJPEG, TJSAMP_422
    #         # yuyv(H,W,2) -> planar Y (H*W), U (H*W/2), V (H*W/2)
    #         yuyv = np.frombuffer(data_bytes, np.uint8).reshape(h, w, 2)
    #         Y = yuyv[:, :, 0]
    #         U = yuyv[:, 0::2, 1]  # Two pixels share one U/V pair
    #         V = yuyv[:, 1::2, 1]
    #         yuv_planar = np.concatenate([Y.reshape(-1), U.reshape(-1), V.reshape(-1)]).tobytes()

    #         jpeg = TurboJPEG()
    #         try:
    #             # New pyTurboJPEG supports quality parameter
    #             return jpeg.encode_yuv(yuv_planar, w, h, TJSAMP_422, quality=quality)
    #         except TypeError:
    #             # Old version doesn't support quality, degrade to default quality
    #             return jpeg.encode_yuv(yuv_planar, w, h, TJSAMP_422)
    #     except Exception:
    #         pass  # Fallback to Branch 2

    # # Branch 2: Other encodings or turbojpeg unavailable -> convert to BGR then imencode
    try:
        bgr = raw_to_bgr(raw)
        if bgr is None:
            return None
        ok, enc_jpg = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return enc_jpg.tobytes() if ok else None
    except Exception:
        return None
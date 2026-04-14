import os
import cv2
import numpy as np

from xrobotoolkit_teleop.utils.dataset.load_data_utils import (
    decode_image,
    get_obs,
    get_all_obs,
    make_mosaic,
    render_trajectory,
    open_writer,
    format_pos_vel_preview,
)

# New: Multi-line auto word-wrap + semi-transparent background text box
def _put_boxed_text(img, text: str, org=(16, 120), max_width=720,
                    font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6,
                    color=(0, 255, 255), thickness=1, line_h=22,
                    bg_color=(0, 0, 0), alpha=0.35):
    if not text:
        return
    x0, y0 = org
    # Word splitting for line wrapping
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip() if cur else w
        (tw, th), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw <= max_width:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    # Background box dimensions
    pad = 8
    w_max = 0
    for ln in lines:
        (tw, _), _ = cv2.getTextSize(ln, font, font_scale, thickness)
        w_max = max(w_max, tw)
    h_box = pad * 2 + line_h * len(lines)
    w_box = pad * 2 + min(max_width, w_max)
    x1, y1 = x0 + w_box, y0 + h_box
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    # Text rendering
    y = y0 + pad + int(line_h * 0.8)
    for ln in lines:
        cv2.putText(img, ln, (x0 + pad, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_h

def play_mujoco_with_preview(
    steps,
    xml_path: str,
    joint_start: int = 0,
    joint_dims: int = 6,
    fps: float = 30.0,
    deg: bool = False,
    window_name: str = "Episode Images",
    save_img_video: str | None = None,
    save_mj_video: str | None = None,
    mj_video_size: tuple[int, int] = (960, 540),
    mj_camera: str | None = None,
    instruction: str | None = None,
):
    """
    Visualize robotic arm using MuJoCo while displaying image info (side+wrist mosaic) in OpenCV window.
    - steps: Steps list of current episode (dataset.pkl structure)
    - xml_path: MuJoCo XML file path
    - joint_start/joint_dims: Start index and dimension of joint vector in state
    - fps: Playback frame rate
    - deg: If joint angles are in degrees, convert to radians; if already radians, keep False
    """
    import time
    try:
        import mujoco
        import mujoco.viewer
    except Exception as e:
        print(f"[player] MuJoCo not available: {e}")
        return
    try:
        # Import path consistent with visualize/data_view.py
        from xrobotoolkit_teleop.utils.dataset.view import RobotArmController
    except Exception as e:
        print(f"[player] import visualize.view.RobotArmController failed: {e}")
        return

    ctrl = RobotArmController(os.path.expanduser(xml_path))
    model, data = ctrl.model, ctrl.data

    # Preprocessing: extract joint vector and image frames from each step
    frames = []
    drop_cnt = 0

    index = 0
    for s in steps:
        # Extract qpos and qvel in one call
        img1, img2, img3, qpos, qvel = get_all_obs(s)
        q = None
        if qpos is not None:
            qpos_arr = np.asarray(qpos, dtype=np.float64).ravel()
            if qpos_arr.size >= (joint_start + joint_dims):
                q = qpos_arr[joint_start:joint_start + joint_dims]
                if deg:
                    q = np.deg2rad(q)
            else:
                drop_cnt += 1
        else:
            drop_cnt += 1
        frames.append((img1, img2, img3, q, qpos, qvel))
        index += 1
    if drop_cnt > 0:
        print(f"[player] warn: {drop_cnt} step(s) have invalid/short state; joints will be held.")

    viewer = mujoco.viewer.launch_passive(model, data)

    # OpenCV preview window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 480)
    img_writer = None  # Lazy creation

    # MuJoCo offscreen rendering (save video)
    
    mj_writer = None
    mj_renderer = None
    if save_mj_video:
        try:
            from mujoco import Renderer
            w, h = mj_video_size
            mj_renderer = Renderer(model, width=w, height=h)
            mj_writer, save_mj_video = open_writer(save_mj_video, fps, (w, h))
            if mj_writer is None:
                print(f"[player] ERROR: cannot open MJ video writer for {save_mj_video}")
            else:
                print(f"[player] MJ video -> {save_mj_video} size={w}x{h} fps={fps}")
        except Exception as e:
            print(f"[player] cannot init MuJoCo offscreen renderer: {e}")
            mj_renderer = None

    period = 1.0 / max(fps, 1.0)
    i = 0
    T = len(frames)
    paused = False
    t0 = time.time()

    recording_mode = bool(save_img_video or save_mj_video)   # Currently writing video
    prev_i = -1                                              # Only write when frame advances

    print(f"[player] MuJoCo + Image replay start: T={T}, fps={fps}, joints={joint_dims} (start={joint_start}, deg={deg})")
    while viewer.is_running():
        # Keyboard control (use non-blocking read even when paused)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (81, ord('a')):  # Left
            i = max(i - 1, 0)
        elif key in (83, ord('d')):  # Right
            i = min(i + 1, T - 1)

        # Current frame
        img1, img2, img3, q, qpos, qvel = frames[i]
        if q is not None:
            data.qpos[ctrl.joint_ids] = q
        mujoco.mj_forward(model, data)
        viewer.sync()

        # Offscreen render MuJoCo frame and write video
        if mj_renderer is not None and mj_writer is not None:
            try:
                mj_renderer.update_scene(data, camera=mj_camera)
                ret = mj_renderer.render()
                if hasattr(mj_renderer, "read_pixels"):
                    rgb = mj_renderer.read_pixels()          # RGB uint8 (H,W,3)
                elif isinstance(ret, np.ndarray):
                    rgb = ret                                 # Some versions return image directly from render()
                else:
                    raise AttributeError("Renderer has neither read_pixels() nor image return from render()")
                mj_writer.write(rgb[..., ::-1])               # RGB->BGR
            except Exception as e:
                print(f"[player] mj render error: {e}")
                mj_writer = None

        # Display and mosaic
        f1 = decode_image(img1) if isinstance(img1, (bytes, np.ndarray)) and img1 is not None else None
        f2 = decode_image(img2) if isinstance(img2, (bytes, np.ndarray)) and img2 is not None else None
        f3 = decode_image(img3) if isinstance(img3, (bytes, np.ndarray)) and img3 is not None else None
       
        mosaic = make_mosaic(f1, f2, f3, target_h=480)
        if mosaic is None:
            mosaic = np.full((480, 640, 3), 0, np.uint8)
        cv2.putText(mosaic, f"step {i+1}/{T}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # New: Display q and dq simultaneously, as well as gripper state
        if qpos is not None or qvel is not None:
            pos_txt, vel_txt, dpos, dvel, grip = format_pos_vel_preview(qpos, qvel, max_items=7)
            cv2.putText(mosaic, f"q[{dpos}]: {pos_txt}{' ...' if dpos>7 else ''}",
                        (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(mosaic, f"dq[{dvel}]: {vel_txt}{' ...' if dvel>7 else ''} | gripper: {grip}",
                        (16, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

        if instruction:
            _put_boxed_text(mosaic, f"Task: {instruction}", org=(16, 120), max_width=900)

        cv2.imshow(window_name, mosaic)

        # Write video frame only when frame index changes; lazy create writer on first write
        if prev_i != i:
            if save_img_video:
                if img_writer is None:
                    h, w = mosaic.shape[:2]
                    img_writer, save_img_video = open_writer(save_img_video, fps, (w, h))
                    if img_writer is None:
                        print(f"[player] ERROR: cannot open video writer for {save_img_video}")
                    else:
                        print(f"[player] IMG video -> {save_img_video} size={w}x{h} fps={fps}")
                if img_writer is not None:
                    img_writer.write(mosaic)

            if mj_renderer is not None and mj_writer is not None:
                try:
                    mj_renderer.update_scene(data, camera=mj_camera)
                    ret = mj_renderer.render()
                    if hasattr(mj_renderer, "read_pixels"):
                        rgb = mj_renderer.read_pixels()
                    elif isinstance(ret, np.ndarray):
                        rgb = ret
                    else:
                        raise AttributeError("Renderer has neither read_pixels() nor image return from render()")
                    mj_writer.write(rgb[..., ::-1])
                except Exception as e:
                    print(f"[player] mj render error: {e}")
                    mj_writer = None
            prev_i = i

        # Playback timing and termination condition
        if not paused:
            target = t0 + (i + 1) * period
            now = time.time()
            if target > now:
                time.sleep(target - now)
            i += 1
            if i >= T:
                if recording_mode:
                    break
                i = T - 1

    cv2.destroyWindow(window_name)
    viewer.close()
    if img_writer is not None: img_writer.release()
    if mj_writer is not None: mj_writer.release()
    print("[player] MuJoCo + Image replay done.")

def build_traj_image(steps, show_traj: bool, window_name: str = "State Trajectory"):
    if not show_traj:
        return None
    # Precompute state trajectory image for entire segment
    states_list = []
    for s in steps:
        _, _, _, st = get_obs(s)
        if st is not None:
            st = np.asarray(st).ravel()
        states_list.append(st)
    # Align different dimensions: stack using minimum dimension
    valid_states = []
    for x in states_list:
        # xn = coerce_numeric(x)
        xn = x
        if xn is not None and xn.size > 0:
            valid_states.append(xn)
    if len(valid_states) == 0:
        return None
    dims = [v.size for v in valid_states]
    target_dim = int(min(dims))
    states_mat = np.stack([v[:target_dim] for v in valid_states], axis=0)  # (T, D)
    traj_img = render_trajectory(states_mat, size=(900, 280), max_dims=6)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, traj_img)
    cv2.resizeWindow(window_name, 900, 280)
    return traj_img


def play_opencv_with_controls(
    steps, fps: float, traj_img=None, 
    window_name: str = "Episode Playback", 
    save_img_video: str | None = None,
    instruction: str | None = None
    ):
    
    T = len(steps)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 480)
    print(f"[player] Loaded episode with {T} steps. Controls: SPACE=Pause/Play, <-/->=Step, +/-=FPS, ESC/Q=Quit")
    paused = False
    i = 0
    delay_ms = max(int(1000.0 / max(fps, 1e-3)), 1)
    writer = None

    while True:
        step = steps[i]
        # Extract qpos + qvel
        img1, img2, img3, qpos, qvel = get_all_obs(step)
        # Support ndarray(BGR, HxWx3) or bytes(jpg/png)
        frame1 = decode_image(img1) if img1 is not None else None
        frame2 = decode_image(img2) if img2 is not None else None
        frame3 = decode_image(img3) if img3 is not None else None

        mosaic = make_mosaic(frame1, frame2, frame3, target_h=480)
        if mosaic is None:
            mosaic = np.full((480, 640, 3), 0, np.uint8)
        cv2.putText(mosaic, f"step {i+1}/{T}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # New: q + dq + gripper state
        if qpos is not None or qvel is not None:
            pos_txt, vel_txt, dpos, dvel, grip = format_pos_vel_preview(qpos, qvel, max_items=7)
            cv2.putText(mosaic, f"q[{dpos}]: {pos_txt}{' ...' if dpos>7 else ''}",
                        (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(mosaic, f"dq[{dvel}]: {vel_txt}{' ...' if dvel>7 else ''} | gripper: {grip}",
                        (16, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

        if instruction:
            _put_boxed_text(mosaic, f"Task: {instruction}", org=(16, 120), max_width=900)

        cv2.imshow(window_name, mosaic)

        if traj_img is not None:
            cv2.imshow("State Trajectory", traj_img)
        if save_img_video:
            if writer is None:
                h, w = mosaic.shape[:2]
                writer, save_img_video = open_writer(save_img_video, fps, (w, h))
                if writer is None:
                    print(f"[player] ERROR: cannot open video writer for {save_img_video}")
                else:
                    print(f"[player] IMG video -> {save_img_video} size={w}x{h} fps={fps}")
            writer.write(mosaic)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key in (27, ord('q'), ord('Q')):  # ESC/Q
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
        elif key in (81, ord('a')):  # Left / 'a'
            i = max(i - 1, 0); continue
        elif key in (83, ord('d')):  # Right / 'd'
            i = min(i + 1, T - 1); continue
        elif key in (ord('+'), ord('=')):
            fps = min(fps + 1, 120); delay_ms = max(int(1000.0 / fps), 1); print(f"[player] fps -> {fps}")
        elif key in (ord('-'), ord('_')):
            fps = max(fps - 1, 1); delay_ms = max(int(1000.0 / fps), 1); print(f"[player] fps -> {fps}")
        if not paused:
            cv2.waitKey(delay_ms)
            i += 1
            if i >= T:
                i = T - 1  # Stop at last frame after playback completes
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
"""
ArmFrankaIncController – incremental (delta-pose) teleoperation controller for the Franka FR3.

Architecture
------------
This controller is adapted from ArmRealManIncController (Quest3-Teleoperation), preserving:
  - XRoboToolkit SDK input pipeline (XrClient)
  - Frame-to-frame incremental EE-target update logic (body-frame right-multiply)
  - Episode/data-collection state machine (idle → homing → teleop)
  - Ros2CollectorClient for /robot_state publication

The Realman-specific parts are replaced:
  - Placo IK + servo_joints()  →  direct Cartesian pose command to franka_pose_cmd_client
  - RealManController          →  FrankaRobotInterface
  - 6-DOF joint state          →  7-DOF Franka joint state

Control Flow
------------
1. XrClient reads Meta Quest 3 right-controller pose at ~100 Hz.
2. Right grip > threshold   → clutch engaged: compute frame-to-frame delta, accumulate into
   ee_target_T (4×4 body-frame right-multiply).
3. Workspace clipping is applied to the translation component of ee_target_T.
4. Per-step velocity limiting prevents large jumps (max delta ~5 mm/step at 100 Hz).
5. ee_target_T is sent directly to the Franka via UDP (franka_pose_cmd_client).
6. Right trigger (0→1) → gripper open/close.

Coordinate Frames
-----------------
  XRoboToolkit SDK  →  R_headset_world  →  robot base frame (Franka: x-forward, y-left, z-up)

  R_headset_world is imported from utils/geometry.py.  It maps the headset's local frame into
  the robot world frame.  Adjust this matrix if the headset is mounted differently relative to
  the Franka base.  See TUNING_GUIDE at the bottom of this file.

Quaternion Conventions
----------------------
  meshcat.transformations (tf): [w, x, y, z]  — used INTERNALLY for incremental updates.
  SciPy / Franka command:       [x, y, z, w]  — used when SENDING commands to the robot.
  State stream from robot:      [x, y, z, w]  — same as SciPy.

  XRT SDK xr_pose layout:  [tx, ty, tz, qx, qy, qz, qw]
    → position indices 0:3
    → quaternion indices 3:7  (qx=3, qy=4, qz=5, qw=6)
    → when converting to tf [w,x,y,z]: [xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]]
"""

import threading
import time
from typing import Optional

import meshcat.transformations as tf
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.franka_robot import (
    DEFAULT_CMD_PORT_FULL,
    DEFAULT_FRANKA_IP,
    DEFAULT_STATE_PORT,
    FrankaRobotInterface,
)
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.ros2_command import Ros2CollectorClient

# ---------------------------------------------------------------------------
# Controller constants (mirrors realman_robots.py structure)
# ---------------------------------------------------------------------------
CONTROLLER_DEADZONE = 0.1   # grip must exceed (1 - deadzone) to activate

# Per-step velocity limits (at 100 Hz):
#   5 mm/step × 100 Hz = 0.5 m/s max translational speed
#   0.05 rad/step × 100 Hz ≈ 5 rad/s max rotational speed
MAX_DELTA_POS_PER_STEP = 0.005   # metres per control step
MAX_DELTA_ROT_PER_STEP = 0.05    # radians per control step

# Homing motion
HOMING_STEP_SIZE = 0.005    # metres/step toward home position (~0.5 m/s at 100 Hz)
HOMING_TOLERANCE_POS = 0.02  # metres – switch to teleop when within this distance
HOME_STEP_ORIENT_SLERP = 0.05  # fraction per step for orientation blending

# Franka reachable workspace (robot base frame).
# These are conservative defaults; tune to your specific setup and task.
# x: forward reach,  y: lateral,  z: height above floor
WORKSPACE_BOUNDS = {
    "x": (0.20, 0.75),
    "y": (-0.45, 0.45),
    "z": (0.08, 0.80),
}

# Default home EE pose (Franka base frame).
# Position: 0.5 m forward, centred, 0.4 m height.
# Orientation: identity quaternion [qx,qy,qz,qw] – aligned with base frame.
# DEFAULT_HOME_POS = np.array([0.5, 0.0, 0.4])
# DEFAULT_HOME_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


DEFAULT_HOME_POS = np.array([0.3, 0.0, 0.59])
DEFAULT_HOME_QUAT_XYZW = np.array([1, 0, 0, 0])


# Default scale: XR controller motion → robot EE motion (1.0 = 1:1)
DEFAULT_SCALE_FACTOR = 1.0

# Quest 3 button mapping:
#   Right grip (hold)         → clutch: robot follows hand
#   Right trigger (analog)    → gripper toggle (press = close, release = open)
#   Right B button            → start episode
#   Right A button            → stop & save episode
DEFAULT_MANIPULATOR_CONFIG = {
    "right_arm": {
        "pose_source": "right_controller",   # XrClient key
        "control_trigger": "right_grip",     # hold right grip to engage clutch
        "gripper_trigger": "right_trigger",  # right trigger controls gripper
        "start_button": "B",                 # right B button: start episode
        "stop_button": "A",                  # right A button: stop episode
        "threshold": 0.8,                    # button threshold (digital: 1.0 when pressed)
        "debounce_s": 0.3,
    },
}


# ---------------------------------------------------------------------------
# Helper: build 4×4 homogeneous matrix from position + quaternion [x,y,z,w]
# ---------------------------------------------------------------------------
def _pose_to_T(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position
    return T


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------
class ArmFrankaIncController:
    """
    Incremental Cartesian teleoperation controller for the Franka FR3.

    Parameters
    ----------
    xr_client : XrClient
        Shared XRoboToolkit SDK wrapper (call XrClient() before instantiating).
    robot_ip : str
        IP of the Franka robot PC.
    robot_cmd_port : int
        UDP command port (default 8888 for pose + gripper command).
    robot_state_port : int
        UDP state-receive port (default 9093).
    home_pos : np.ndarray (3,)
        Home EE position in robot base frame [m].
    home_quat_xyzw : np.ndarray (4,)
        Home EE orientation [qx, qy, qz, qw].
    R_headset_world : np.ndarray (3,3)
        Rotation from XRT headset frame to robot world frame.
    scale_factor : float
        Scale applied to XR controller translation deltas (1.0 = 1:1).
    manipulator_config : dict
        Per-arm XrClient key mapping (see DEFAULT_MANIPULATOR_CONFIG).
    collect_change_threshold : float
        Minimum gripper trigger change to consider robot "active" for data collection.
    collect_idle_timeout : float
        Seconds of inactivity before data collection is paused.
    publish_state_mode : str
        "internal" – /robot_state published from this process.
        "external" – robot_pub node publishes state on command.
    dry_run : bool
        If True, pose commands are printed instead of sent.  Safe for frame-check tests.
    """

    def __init__(
        self,
        xr_client: XrClient,
        robot_ip: str = DEFAULT_FRANKA_IP,
        robot_cmd_port: int = DEFAULT_CMD_PORT_FULL,
        robot_state_port: int = DEFAULT_STATE_PORT,
        home_pos: np.ndarray = DEFAULT_HOME_POS,
        home_quat_xyzw: np.ndarray = DEFAULT_HOME_QUAT_XYZW,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        manipulator_config: dict = DEFAULT_MANIPULATOR_CONFIG,
        collect_change_threshold: float = 0.01,
        collect_idle_timeout: float = 0.5,
        publish_state_mode: str = "internal",
        dry_run: bool = False,
    ):
        self.xr_client = xr_client
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.manipulator_config = manipulator_config
        self.dry_run = dry_run

        # Home pose (used for episode reset / homing)
        self._home_pos = np.asarray(home_pos, dtype=float).reshape(3)
        self._home_quat_xyzw = np.asarray(home_quat_xyzw, dtype=float).reshape(4)
        # Normalise
        n = np.linalg.norm(self._home_quat_xyzw)
        if n > 1e-12:
            self._home_quat_xyzw /= n

        # Franka robot interface
        self.robot = FrankaRobotInterface(
            ip=robot_ip,
            cmd_port=robot_cmd_port,
            state_port=robot_state_port,
            dry_run=dry_run,
        )

        # ---------------------------------------------------------------
        # Incremental EE-target state (one entry per arm in manipulator_config)
        # ---------------------------------------------------------------
        # ee_target_T: persistent 4×4 world-frame target for each arm.
        # None means the arm has not been activated yet in this episode.
        self.ee_target_T: dict[str, Optional[np.ndarray]] = {name: None for name in manipulator_config}

        # XR controller reference poses (frame-to-frame delta computation).
        # Reset to None on activate/deactivate edge to latch a new reference.
        self.init_controller_xyz: dict[str, Optional[np.ndarray]] = {name: None for name in manipulator_config}
        self.init_controller_quat: dict[str, Optional[np.ndarray]] = {name: None for name in manipulator_config}

        # Previous activation state (edge detection)
        self._prev_active: dict[str, bool] = {name: False for name in manipulator_config}

        # ---------------------------------------------------------------
        # Gripper state
        # ---------------------------------------------------------------
        self.gripper_trigger_val: dict[str, float] = {name: 0.0 for name in manipulator_config}

        # ---------------------------------------------------------------
        # Episode / mode state machine
        # idle    : robot at rest; no commands sent
        # homing  : robot returning to home pose; IK thread guides it there
        # teleop  : robot follows XR controller
        # ---------------------------------------------------------------
        self.mode = "idle"
        self.episode_collecting = False
        self._homing_started = False

        # Button edge-detection state
        self._ep_prev_start = 0
        self._ep_prev_stop = 0
        self._ep_last_ns = 0

        # ---------------------------------------------------------------
        # Data collection
        # ---------------------------------------------------------------
        self.collect_change_threshold = collect_change_threshold
        self.collect_idle_timeout = collect_idle_timeout

        self.publish_state_mode = (publish_state_mode or "internal").lower()
        if self.publish_state_mode not in ("internal", "external"):
            print(f"[Franka] Unknown publish_state_mode={self.publish_state_mode}; fallback to 'internal'")
            self.publish_state_mode = "internal"

        self.collector_client = Ros2CollectorClient(mode=self.publish_state_mode)
        if self.publish_state_mode == "internal":
            self.collector_client.set_state_provider(
                self._state_provider,
                topic="/robot_state",
                rate_hz=10.0,
                reliable=True,
            )

        self.prev_gripper_val: dict[str, float] = {name: 0.0 for name in manipulator_config}
        self.collecting: dict[str, bool] = {name: False for name in manipulator_config}
        self.last_activity_ts: dict[str, float] = {name: 0.0 for name in manipulator_config}

        # For joint-velocity estimation in state provider
        self._js_prev_q: Optional[list] = None
        self._js_prev_ns: Optional[int] = None

        if dry_run:
            print("[Franka] DRY RUN MODE – commands printed, not sent to robot.")

    # ===================================================================
    # XR pose processing (body-frame incremental delta)
    # ===================================================================

    def _process_xr_pose(
        self, xr_pose: np.ndarray, arm_name: str, R_world_ee: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the frame-to-frame delta from the XR controller.

        The XRT SDK returns poses as [tx, ty, tz, qx, qy, qz, qw] in the
        headset's local coordinate frame.  We:
          1. Transform position and orientation to the robot world frame.
          2. Compute the delta from the previously stored reference pose.
          3. Update the reference pose (so each call yields one-step delta).
          4. Rotate the delta from the world frame into the current EE local frame.

        Returns
        -------
        delta_xyz_local : (3,)  positional delta in the current actual-EE frame [m]
        delta_rot_local : (3,)  rotation delta as angle-axis in the current actual-EE frame [rad]
        """
        # XRT SDK layout: [tx, ty, tz, qx, qy, qz, qw]
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        # Convert to meshcat [w,x,y,z] format for tf operations
        controller_quat_wxyz = np.array([xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]])

        # --- Step 1: Rotate from headset frame to world frame ---
        # Position
        controller_xyz = self.R_headset_world @ controller_xyz

        # Orientation: conjugate-multiply R_quat on both sides
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)  # [w,x,y,z]
        controller_quat_wxyz = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat_wxyz),
            tf.quaternion_conjugate(R_quat),
        )

        # --- Step 2: Latch reference on first call after (re-)activation ---
        if self.init_controller_xyz[arm_name] is None:
            self.init_controller_xyz[arm_name] = controller_xyz.copy()
            self.init_controller_quat[arm_name] = controller_quat_wxyz.copy()
            return np.zeros(3), np.zeros(3)  # Zero delta on first frame

        # --- Step 3: Frame-to-frame delta in world frame ---
        delta_xyz_world = (controller_xyz - self.init_controller_xyz[arm_name]) * self.scale_factor
        delta_rot_world = quat_diff_as_angle_axis(self.init_controller_quat[arm_name], controller_quat_wxyz)

        # Update reference to current frame (so next call gives the next delta)
        self.init_controller_xyz[arm_name] = controller_xyz.copy()
        self.init_controller_quat[arm_name] = controller_quat_wxyz.copy()

        # --- Step 4: Rotate world-frame delta into current EE local frame ---
        # R_ee_world = R_world_ee.T  (transpose = inverse for rotation matrix)
        R_ee_world = R_world_ee.T
        delta_xyz_local = R_ee_world @ delta_xyz_world
        delta_rot_local = R_ee_world @ delta_rot_world

        return delta_xyz_local, delta_rot_local

    # ===================================================================
    # Safety helpers
    # ===================================================================

    @staticmethod
    def _clip_delta_pos(delta: np.ndarray) -> np.ndarray:
        """Clip translational delta to MAX_DELTA_POS_PER_STEP per axis."""
        return np.clip(delta, -MAX_DELTA_POS_PER_STEP, MAX_DELTA_POS_PER_STEP)

    @staticmethod
    def _clip_delta_rot(delta: np.ndarray) -> np.ndarray:
        """Clip rotational delta to MAX_DELTA_ROT_PER_STEP per step."""
        mag = np.linalg.norm(delta)
        if mag > MAX_DELTA_ROT_PER_STEP:
            return delta * (MAX_DELTA_ROT_PER_STEP / mag)
        return delta

    @staticmethod
    def _clip_workspace(pos: np.ndarray) -> np.ndarray:
        """Clip EE target position to the defined workspace bounding box."""
        return np.array([
            np.clip(pos[0], WORKSPACE_BOUNDS["x"][0], WORKSPACE_BOUNDS["x"][1]),
            np.clip(pos[1], WORKSPACE_BOUNDS["y"][0], WORKSPACE_BOUNDS["y"][1]),
            np.clip(pos[2], WORKSPACE_BOUNDS["z"][0], WORKSPACE_BOUNDS["z"][1]),
        ])

    # ===================================================================
    # Homing
    # ===================================================================

    def _homing_step(self) -> None:
        """
        Gradually move the EE target toward the home pose.

        On first call (self._homing_started == False), initialises ee_target_T
        from the current actual EE pose to avoid a sudden jump.
        Switches mode to "teleop" once the target is within HOMING_TOLERANCE_POS of home.
        """
        if not self._homing_started:
            # Initialise EE target from current actual pose (no sudden jump)
            ee = self.robot.get_end_effector_pose()
            if self.robot.is_state_available():
                T_init = _pose_to_T(ee.translation, ee.quaternion)
            else:
                # No state stream: start from home directly
                T_init = _pose_to_T(self._home_pos, self._home_quat_xyzw)

            for arm_name in self.manipulator_config:
                self.ee_target_T[arm_name] = T_init.copy()
                self.init_controller_xyz[arm_name] = None
                self.init_controller_quat[arm_name] = None
                self._prev_active[arm_name] = False

            self._homing_started = True
            print("[Franka] Homing started – moving to home pose...")

        # Build home transform
        T_home = _pose_to_T(self._home_pos, self._home_quat_xyzw)

        for arm_name in self.manipulator_config:
            if self.ee_target_T[arm_name] is None:
                self.ee_target_T[arm_name] = T_home.copy()
                continue

            current_pos = self.ee_target_T[arm_name][:3, 3]
            err_pos = np.linalg.norm(current_pos - self._home_pos)

            if err_pos > HOMING_TOLERANCE_POS:
                # Step toward home
                direction = self._home_pos - current_pos
                step = min(err_pos, HOMING_STEP_SIZE)
                new_pos = current_pos + (direction / err_pos) * step
                self.ee_target_T[arm_name][:3, 3] = new_pos

                # Blend orientation toward home (SLERP fraction per step)
                current_quat_xyzw = Rotation.from_matrix(self.ee_target_T[arm_name][:3, :3]).as_quat()
                key_rots = Rotation.from_quat([current_quat_xyzw, self._home_quat_xyzw])
                slerp = Slerp([0.0, 1.0], key_rots)
                self.ee_target_T[arm_name][:3, :3] = slerp(HOME_STEP_ORIENT_SLERP).as_matrix()
            else:
                # Close enough – snap to home orientation
                self.ee_target_T[arm_name][:3, 3] = self._home_pos.copy()
                self.ee_target_T[arm_name][:3, :3] = Rotation.from_quat(self._home_quat_xyzw).as_matrix()

        # Check convergence on the first arm (single-arm setup)
        primary_arm = next(iter(self.manipulator_config))
        if self.ee_target_T[primary_arm] is not None:
            err = np.linalg.norm(self.ee_target_T[primary_arm][:3, 3] - self._home_pos)
            if err <= HOMING_TOLERANCE_POS:
                self.mode = "teleop"
                print("[Franka] Homing complete – teleop enabled.")

        # Send current homing target to robot
        self._send_all_targets()

    # ===================================================================
    # Target computation and sending  (teleop mode core)
    # ===================================================================

    def calc_target_and_send(self) -> None:
        """
        Main per-step function (called at control rate, typically 100 Hz).

        In idle mode: does nothing.
        In homing mode: delegates to _homing_step().
        In teleop mode: reads XR controller, updates ee_target_T, sends to Franka.
        """
        if self.mode == "idle":
            return

        if self.mode == "homing":
            self._homing_step()
            return

        # ---- teleop mode ----
        for arm_name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)

            # Read actual EE pose from robot state stream for reference frame
            ee_actual = self.robot.get_end_effector_pose()
            T_actual = _pose_to_T(ee_actual.translation, ee_actual.quaternion)
            R_actual = T_actual[:3, :3]  # Actual world→EE rotation (used for delta frame)

            was_active = self._prev_active[arm_name]

            # ---- Edge: deactivated → activated ----
            if (not was_active) and active:
                if self.ee_target_T[arm_name] is None:
                    # First activation ever: latch current actual pose as target
                    self.ee_target_T[arm_name] = T_actual.copy()
                # Reset XR reference so first delta is zero (re-clutch)
                self.init_controller_xyz[arm_name] = None
                self.init_controller_quat[arm_name] = None
                print(f"[Franka] {arm_name} clutch ENGAGED. EE at {ee_actual.translation.round(3)}")

            # ---- Edge: activated → deactivated ----
            elif was_active and (not active):
                # Reset XR reference; ee_target_T is kept so robot holds last pose
                self.init_controller_xyz[arm_name] = None
                self.init_controller_quat[arm_name] = None
                print(f"[Franka] {arm_name} clutch DISENGAGED.")

            self._prev_active[arm_name] = active

            # ---- Accumulate delta when clutch is engaged ----
            if active and self.ee_target_T[arm_name] is not None:
                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])

                # Delta in actual EE local frame
                delta_xyz_local_actual, delta_rot_local_actual = self._process_xr_pose(
                    xr_pose, arm_name, R_actual
                )

                # Convert from actual EE frame to target EE frame.
                # This avoids drift when actual and target orientations differ.
                R_target = self.ee_target_T[arm_name][:3, :3]
                R_targ_cur = R_target.T @ R_actual
                delta_xyz_des = R_targ_cur @ delta_xyz_local_actual
                delta_rot_des = R_targ_cur @ delta_rot_local_actual

                # Per-step velocity limiting (safety)
                delta_xyz_des = self._clip_delta_pos(delta_xyz_des)
                delta_rot_des = self._clip_delta_rot(delta_rot_des)

                # Build increment as 4×4 transform (body frame)
                theta = np.linalg.norm(delta_rot_des)
                if theta > 1e-9:
                    axis = delta_rot_des / theta
                    q_inc_wxyz = tf.quaternion_about_axis(theta, axis)  # [w,x,y,z]
                    R_inc = tf.quaternion_matrix(q_inc_wxyz)[:3, :3]
                else:
                    R_inc = np.eye(3)

                T_inc = np.eye(4)
                T_inc[:3, :3] = R_inc
                T_inc[:3, 3] = delta_xyz_des

                # Right-multiply: apply increment in body (EE target) frame
                self.ee_target_T[arm_name] = self.ee_target_T[arm_name] @ T_inc

                # Workspace clipping on the world-frame position
                self.ee_target_T[arm_name][:3, 3] = self._clip_workspace(
                    self.ee_target_T[arm_name][:3, 3]
                )

        # Send commands for all arms (hold last target when clutch is off)
        self._send_all_targets()

    def _send_all_targets(self) -> None:
        """Extract pose from ee_target_T and send to Franka for each configured arm."""
        for arm_name in self.manipulator_config:
            if self.ee_target_T[arm_name] is None:
                continue

            pos = self.ee_target_T[arm_name][:3, 3]
            # Convert rotation matrix → [qx, qy, qz, qw] (SciPy / Franka convention)
            quat_xyzw = Rotation.from_matrix(self.ee_target_T[arm_name][:3, :3]).as_quat()

            # Map trigger (0→1) to binary gripper command (threshold at 0.5)
            gripper_btn = 1.0 if self.gripper_trigger_val.get(arm_name, 0.0) > 0.5 else 0.0

            self.robot.send_eef_pose_command(pos, quat_xyzw, gripper_btn=gripper_btn)

    # ===================================================================
    # Gripper control thread body
    # ===================================================================

    def control_gripper(self) -> None:
        """Read XR trigger values into gripper_trigger_val for next send cycle."""
        for arm_name, config in self.manipulator_config.items():
            self.gripper_trigger_val[arm_name] = self.xr_client.get_key_value_by_name(config["gripper_trigger"])

    # ===================================================================
    # Data collection helpers
    # ===================================================================

    def _state_provider(self) -> dict:
        """
        Build the JointState dict published to /robot_state at 10 Hz.

        Field layout (matches episode_recorder.py expectations):
          joint1..joint7   : 7-DOF positions [rad]
          gripper          : right trigger value (normalised 0-1)
          ee_x..ee_qw      : EE position [m] and orientation [qx,qy,qz,qw]
          joint1_vel..joint7_vel : joint velocities [rad/s]
        """
        q = self.robot.get_joint_positions()   # (7,) rad, may contain NaN
        dq = self.robot.get_joint_velocities()  # (7,) rad/s
        ee = self.robot.get_end_effector_pose()

        # Use zeros where state is unavailable (NaN-safe)
        q_safe = np.nan_to_num(q, nan=0.0).reshape(7).tolist()
        dq_safe = np.nan_to_num(dq, nan=0.0).reshape(7).tolist()

        gripper_val = self.gripper_trigger_val.get(next(iter(self.manipulator_config)), 0.0)

        now_ns = time.time_ns()
        if self._js_prev_q is not None and self._js_prev_ns is not None and now_ns > self._js_prev_ns:
            dt = (now_ns - self._js_prev_ns) / 1e9
            if dt > 1e-6:
                dq_estimated = [(q_safe[i] - self._js_prev_q[i]) / dt for i in range(7)]
            else:
                dq_estimated = dq_safe
        else:
            dq_estimated = dq_safe

        self._js_prev_q = q_safe
        self._js_prev_ns = now_ns

        names = (
            [f"joint{i+1}" for i in range(7)]
            + ["gripper"]
            + ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
            + [f"joint{i+1}_vel" for i in range(7)]
        )
        positions = (
            q_safe
            + [gripper_val]
            + ee.translation.tolist()
            + ee.quaternion.tolist()
            + dq_estimated
        )

        return {"name": names, "position": positions, "frame_id": "base_link"}

    def update_collect_state(self) -> None:
        """
        Start/stop the Ros2CollectorClient timer based on robot activity.
        Activity is defined as: clutch engaged OR gripper trigger changed significantly.
        """
        for arm_name in self.manipulator_config:
            prev = self.prev_gripper_val[arm_name]
            cur = self.gripper_trigger_val.get(arm_name, 0.0)
            changed = abs(cur - prev) > self.collect_change_threshold
            self.prev_gripper_val[arm_name] = cur

            now = time.time()
            grip_active = self._prev_active.get(arm_name, False)

            if grip_active or changed:
                self.last_activity_ts[arm_name] = now
                if not self.collecting[arm_name]:
                    self.collecting[arm_name] = True
                    try:
                        if self.publish_state_mode == "internal":
                            self.collector_client.start_collect()
                        else:
                            self.collector_client.start_collect_cmd()
                    except Exception as exc:
                        print(f"[Franka][Collector] start error: {exc}")
            else:
                if self.collecting[arm_name] and (now - self.last_activity_ts[arm_name] >= self.collect_idle_timeout):
                    self.collecting[arm_name] = False
                    try:
                        if self.publish_state_mode == "internal":
                            self.collector_client.stop_collect()
                        else:
                            self.collector_client.stop_collect_cmd()
                    except Exception as exc:
                        print(f"[Franka][Collector] stop error: {exc}")

    # ===================================================================
    # Episode control (left grip = start, left trigger = stop)
    # ===================================================================

    # UDP port that run_collection.py listens on for START/STOP signals.
    COLLECTION_CTRL_PORT = 8765

    def _publish_episode_cmd_async(self, cmd: str) -> None:
        def _worker():
            try:
                if cmd == "START":
                    self.collector_client.start_episode()
                elif cmd == "STOP":
                    self.collector_client.stop_episode()
            except Exception as exc:
                print(f"[Franka][Episode] publish '{cmd}' error: {exc}")
            # Also broadcast to run_collection.py via local UDP
            try:
                import socket as _socket
                with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as _s:
                    _s.sendto(cmd.encode(), ("127.0.0.1", ArmFrankaIncController.COLLECTION_CTRL_PORT))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    @staticmethod
    def _edge_pressed(cur: float, prev: int, threshold: float) -> bool:
        return (cur >= threshold) and (prev == 0)

    def update_episode_control(self) -> None:
        """
        Monitor right B (start) and right A (stop) for episode control.
        On START: enters homing mode (robot returns to home, then switches to teleop).
        On STOP:  publishes STOP signal and returns to teleop (continues until next START).
        """
        cfg = self.manipulator_config[next(iter(self.manipulator_config))]
        v_start = self.xr_client.get_key_value_by_name(cfg["start_button"])
        v_stop = self.xr_client.get_key_value_by_name(cfg["stop_button"])

        now_ns = time.time_ns()
        debounce_ok = (now_ns - self._ep_last_ns) >= int(cfg["debounce_s"] * 1e9)

        if (
            debounce_ok
            and not self.episode_collecting
            and self._edge_pressed(v_start, self._ep_prev_start, cfg["threshold"])
        ):
            self.episode_collecting = True
            self._ep_last_ns = now_ns
            # Enter homing: robot moves to home pose, then switches to teleop
            self.mode = "homing"
            self._homing_started = False
            self._publish_episode_cmd_async("START")
            print("[Franka] Episode STARTED – homing before teleop.")

        if (
            debounce_ok
            and self.episode_collecting
            and self._edge_pressed(v_stop, self._ep_prev_stop, cfg["threshold"])
        ):
            self.episode_collecting = False
            self._ep_last_ns = now_ns
            self._publish_episode_cmd_async("STOP")
            print("[Franka] Episode STOPPED.")
            # Remain in teleop (or drop to idle if desired)
            self.mode = "teleop"
            self._homing_started = False

        self._ep_prev_start = 1 if v_start >= cfg["threshold"] else 0
        self._ep_prev_stop = 1 if v_stop >= cfg["threshold"] else 0

    # ===================================================================
    # Thread entry points
    # ===================================================================

    def run_ik_and_control_thread(self, stop_event: threading.Event) -> None:
        """
        Main control thread: runs at ~100 Hz.
        Computes and sends EE target on every step.
        Closes robot connection on exit.
        """
        print("[Franka] Control thread started.")
        while not stop_event.is_set():
            try:
                start = time.time()
                self.calc_target_and_send()
                elapsed = time.time() - start
                sleep_time = 0.01 - elapsed  # target 100 Hz
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as exc:
                print(f"[Franka] Control loop error: {exc}")

        print("[Franka] Control thread stopped.")
        self.robot.close()

    def run_gripper_control_thread(self, stop_event: threading.Event) -> None:
        """Reads XR trigger values into gripper_trigger_val at ~1 kHz."""
        print("[Franka] Gripper thread started.")
        while not stop_event.is_set():
            self.control_gripper()
            time.sleep(0.001)

    def run_collect_state_thread(self, stop_event: threading.Event) -> None:
        """Manages Ros2CollectorClient start/stop at 2 Hz."""
        print("[Franka] Collect-state thread started.")
        while not stop_event.is_set():
            self.update_collect_state()
            time.sleep(0.5)

    def run_episode_control_thread(self, stop_event: threading.Event) -> None:
        """Monitors episode start/stop buttons at 100 Hz."""
        print("[Franka] Episode-control thread started.")
        while not stop_event.is_set():
            self.update_episode_control()
            time.sleep(0.01)

    # ===================================================================
    # Public API
    # ===================================================================

    def reset(self) -> None:
        """Trigger a homing sequence (can be called externally)."""
        self.mode = "homing"
        self._homing_started = False

    def set_mode(self, mode: str) -> None:
        """Manually set the controller mode: 'idle' | 'homing' | 'teleop'."""
        assert mode in ("idle", "homing", "teleop"), f"Unknown mode: {mode}"
        if mode == "homing":
            self._homing_started = False
        self.mode = mode
        print(f"[Franka] Mode set to '{mode}'.")

    def close(self) -> None:
        try:
            self.collector_client.close()
        except Exception:
            pass
        try:
            self.robot.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TUNING GUIDE
# ---------------------------------------------------------------------------
# Parameters to adjust for your specific setup (in order of priority):
#
# 1. R_HEADSET_TO_WORLD  (utils/geometry.py or passed to constructor)
#    Maps the XRoboToolkit headset frame to the Franka base frame.
#    Default: [[0,0,1],[1,0,0],[0,1,0]]
#    Verify with dry_run=True: hold controller still, confirm robot EE target does not drift.
#    Then move controller +X: verify robot moves in the expected direction.
#
# 2. scale_factor  (default 1.0)
#    Ratio of robot EE motion to controller motion.
#    Increase (>1) for coarser control; decrease (<1) for fine manipulation.
#    Typical range: 0.5 – 1.5
#
# 3. home_pos / home_quat_xyzw
#    The Cartesian pose the robot returns to at episode start.
#    Set to a pose that is safe and within the reachable workspace.
#
# 4. WORKSPACE_BOUNDS  (module-level dict)
#    Conservative Cartesian bounding box.  Tighten for your specific task area.
#
# 5. MAX_DELTA_POS_PER_STEP / MAX_DELTA_ROT_PER_STEP
#    Per-step velocity limits.  Reduce if robot moves too fast.
#
# COORDINATE FRAME CHECKLIST  (run with dry_run=True)
# -------------------------------------------------------
# [ ] Hold right controller still → ee_target_T does not drift
# [ ] Move controller +X (right)  → robot moves in expected lateral direction
# [ ] Move controller +Y (up)     → robot moves upward
# [ ] Move controller +Z (forward)→ robot moves forward
# [ ] Rotate controller wrist     → robot EE rotates correspondingly
# [ ] Release grip, move hand     → robot holds (clutch disengaged)
# [ ] Re-grip                     → robot continues from current pose (no jump)

"""
FrankaRobotInterface – low-level UDP client for the Franka FR3 robot.

Communicates with ``franka_pose_cmd_client`` running on the robot PC via two UDP channels:

  Command  (PC → robot):  text string sent to ``robot_ip:cmd_port``
  State    (robot → PC):  JSON line received on ``0.0.0.0:state_port``

Command string formats
----------------------
Pose-only (7 values, used when ``include_gripper=False``):
    "x y z qx qy qz qw"

Pose + gripper (13 values, used when ``include_gripper=True``, default port 8888):
    "x y z qx qy qz qw gripper_btn speed force eps_inner eps_outer width"

    gripper_btn  : 1.0 = close, 0.0 = open (one-shot)
    speed        : m/s (default 0.1)
    force        : N   (default 20.0)
    eps_inner/outer : grasp epsilon metres (default 0.04)
    width        : target grasp width metres (default 0.0)

Quaternion convention: **scalar-last** [qx, qy, qz, qw]  (SciPy / ROS convention).
This is the same convention used by franka_r3_robot.py in franka_research.

State JSON keys (received from franka_pose_cmd_client):
    robot0_eef_pos          : [x, y, z]           metres
    robot0_eef_quat         : [qx, qy, qz, qw]    scalar-last
    robot0_joint_pos        : [q1..q7]             radians
    robot0_joint_vel        : [dq1..dq7]           rad/s
    robot0_joint_ext_torque : [tau1..tau7]         N·m
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Default network parameters (mirror franka_research defaults)
# ---------------------------------------------------------------------------
DEFAULT_CMD_PORT_FULL = 8888    # pose + gripper command (vr_to_robot_converter protocol)
DEFAULT_CMD_PORT_POSE = 8890    # pose-only command      (franka_r3_robot protocol)
DEFAULT_STATE_PORT = 9093       # JSON state stream (from robot PC)
DEFAULT_FRANKA_IP = "10.1.38.145"

# Local port to which each received state packet is forwarded so that other
# processes (e.g. run_collection.py) can read state without competing for 9093.
STATE_MIRROR_PORT = 9094

# ---------------------------------------------------------------------------
# Default gripper parameters (Franka Hand)
# ---------------------------------------------------------------------------
DEFAULT_GRIPPER_SPEED = 0.1     # m/s
DEFAULT_GRIPPER_FORCE = 20.0    # N
DEFAULT_GRIPPER_EPS_INNER = 0.04  # m
DEFAULT_GRIPPER_EPS_OUTER = 0.04  # m


@dataclass
class EEFPose:
    """End-effector pose container.  Quaternion is scalar-last [qx, qy, qz, qw]."""

    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))


class FrankaRobotInterface:
    """
    UDP-based interface to the Franka FR3 robot.

    Wraps the same protocol used by franka_r3_robot.py (franka_research codebase)
    and the vr_to_robot_converter.py, extended with gripper control.

    Parameters
    ----------
    ip : str
        IP address of the robot PC running franka_pose_cmd_client.
    cmd_port : int
        UDP port for pose commands.
    state_port : int
        UDP port to listen on for JSON robot-state datagrams (``0.0.0.0:state_port``).
        Set to None to disable state reception.
    include_gripper : bool
        If True, send the 13-value command (pose + gripper state).
        If False, send the 7-value pose-only command.
    connect_timeout : float
        Seconds to wait for the first state packet before proceeding.
    dry_run : bool
        If True, commands are printed to stdout instead of sent over UDP.
        Useful for verifying coordinate frames before moving the real robot.
    """

    def __init__(
        self,
        ip: str = DEFAULT_FRANKA_IP,
        cmd_port: int = DEFAULT_CMD_PORT_FULL,
        state_port: int = DEFAULT_STATE_PORT,
        state_listen_ip: str = "0.0.0.0",
        include_gripper: bool = True,
        connect_timeout: float = 3.0,
        dry_run: bool = False,
    ):
        self.ip = ip
        self.cmd_port = int(cmd_port)
        self.include_gripper = include_gripper
        self.dry_run = dry_run

        # Command socket (send-only UDP)
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_lock = threading.Lock()
        self._last_cmd: Optional[str] = None

        # State receiver thread
        self._state: Optional[Dict] = None
        self._state_lock = threading.Lock()
        self._running = True
        self._state_sock: Optional[socket.socket] = None

        # Mirror socket: forwards each received state packet to STATE_MIRROR_PORT
        # so other local processes can receive state without competing for state_port.
        self._mirror_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if state_port is not None:
            self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            self._state_sock.bind((state_listen_ip, int(state_port)))
            self._state_sock.settimeout(0.2)

            self._state_thread = threading.Thread(
                target=self._state_receiver_loop,
                daemon=True,
                name="franka-state-rx",
            )
            self._state_thread.start()

            # Wait for first state packet to confirm connectivity
            if connect_timeout > 0.0:
                deadline = time.monotonic() + connect_timeout
                while time.monotonic() < deadline:
                    if self._get_state() is not None:
                        print("[FrankaRobotInterface] State stream connected.")
                        break
                    time.sleep(0.05)
                else:
                    print(
                        "[FrankaRobotInterface] WARNING: No state stream received within "
                        f"{connect_timeout:.1f}s. Continuing without state feedback."
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_receiver_loop(self) -> None:
        assert self._state_sock is not None
        while self._running:
            try:
                data, _ = self._state_sock.recvfrom(65536)
                # Forward raw packet to mirror port for other local processes
                try:
                    self._mirror_sock.sendto(data, ("127.0.0.1", STATE_MIRROR_PORT))
                except Exception:
                    pass
                text = data.decode("utf-8", errors="ignore").strip()
                if not text:
                    continue
                # Take the last complete JSON line in the datagram
                line = text.splitlines()[-1]
                state = json.loads(line)
                with self._state_lock:
                    self._state = state
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                pass  # Malformed packet; skip
            except Exception as exc:
                if self._running:
                    print(f"[FrankaRobotInterface] State rx error: {exc}")
                time.sleep(0.02)

    def _get_state(self) -> Optional[Dict]:
        with self._state_lock:
            return dict(self._state) if self._state is not None else None

    @staticmethod
    def _normalize_quat(q: np.ndarray) -> np.ndarray:
        """Normalize to unit quaternion; returns identity [0,0,0,1] if near-zero."""
        n = np.linalg.norm(q)
        return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    # Command interface
    # ------------------------------------------------------------------

    def send_eef_pose_command(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_btn: float = 0.0,
        gripper_speed: float = DEFAULT_GRIPPER_SPEED,
        gripper_force: float = DEFAULT_GRIPPER_FORCE,
        gripper_eps_inner: float = DEFAULT_GRIPPER_EPS_INNER,
        gripper_eps_outer: float = DEFAULT_GRIPPER_EPS_OUTER,
        gripper_width: float = 0.0,
    ) -> None:
        """
        Send an absolute end-effector pose command (and optional gripper state).

        Parameters
        ----------
        position : array-like (3,)
            Target EE position in robot base frame [m].
        quaternion : array-like (4,)
            Target EE orientation as [qx, qy, qz, qw] (scalar-last).
        gripper_btn : float
            1.0 = trigger a close, 0.0 = no gripper action (one-shot semantics).
        """
        pos = np.asarray(position, dtype=float).reshape(3)
        quat = self._normalize_quat(np.asarray(quaternion, dtype=float).reshape(4))

        if self.include_gripper:
            cmd = (
                f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} "
                f"{gripper_btn:.6f} {gripper_speed:.6f} {gripper_force:.6f} "
                f"{gripper_eps_inner:.6f} {gripper_eps_outer:.6f} {gripper_width:.6f}"
            )
        else:
            cmd = (
                f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
            )

        if self.dry_run:
            print(f"[FrankaRobotInterface][DRY RUN] {cmd}")
            return

        with self._cmd_lock:
            self._last_cmd = cmd
            self._cmd_sock.sendto(cmd.encode("utf-8"), (self.ip, self.cmd_port))

    # ------------------------------------------------------------------
    # State readback
    # ------------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Returns 7-DOF joint positions [rad].  NaN-filled if unavailable."""
        state = self._get_state()
        if state and "robot0_joint_pos" in state:
            return np.asarray(state["robot0_joint_pos"], dtype=float).reshape(7)
        return np.full(7, np.nan)

    def get_joint_velocities(self) -> np.ndarray:
        """Returns 7-DOF joint velocities [rad/s].  NaN-filled if unavailable."""
        state = self._get_state()
        if state and "robot0_joint_vel" in state:
            return np.asarray(state["robot0_joint_vel"], dtype=float).reshape(7)
        return np.full(7, np.nan)

    def get_joint_torques(self) -> np.ndarray:
        """Returns 7-DOF external joint torques [N·m].  NaN-filled if unavailable."""
        state = self._get_state()
        if state and "robot0_joint_ext_torque" in state:
            return np.asarray(state["robot0_joint_ext_torque"], dtype=float).reshape(7)
        return np.full(7, np.nan)

    def get_end_effector_pose(self) -> EEFPose:
        """Returns current EE pose from state stream. Falls back to zeros if unavailable."""
        state = self._get_state()
        if state and "robot0_eef_pos" in state and "robot0_eef_quat" in state:
            pos = np.asarray(state["robot0_eef_pos"], dtype=float).reshape(3)
            quat = self._normalize_quat(np.asarray(state["robot0_eef_quat"], dtype=float).reshape(4))
            return EEFPose(translation=pos, quaternion=quat)
        return EEFPose()

    def is_state_available(self) -> bool:
        """True if at least one state packet has been received."""
        return self._get_state() is not None

    # ------------------------------------------------------------------
    # Compatibility stubs (mirrors Realman interface API)
    # ------------------------------------------------------------------

    def get_current_joint_positions(self) -> np.ndarray:
        """Alias for get_joint_positions()."""
        return self.get_joint_positions()

    def get_gripper_state(self):
        """
        Returns (gripper_open_estimate, speed).
        Gripper state is not in the Franka state stream; returns defaults.
        """
        return 1.0, 0.0  # Assume open; speed unknown

    def get_gripper_open_position(self) -> float:
        return 0.0  # 0.0 = fully open in normalised [0, 1] trigger space

    def get_gripper_close_position(self) -> float:
        return 1.0  # 1.0 = fully closed

    def reset(self) -> None:
        """
        Stub: physical reset (joint-space home) is handled by the robot-side
        franka_pose_cmd_client.  Call FrankaRobotInterface.send_eef_pose_command
        with the home pose to achieve Cartesian homing.
        """
        print("[FrankaRobotInterface] reset() stub – use Cartesian homing from controller.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._running = False
        try:
            self._cmd_sock.close()
        except Exception:
            pass
        if self._state_sock is not None:
            try:
                self._state_sock.close()
            except Exception:
                pass
        try:
            self._mirror_sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

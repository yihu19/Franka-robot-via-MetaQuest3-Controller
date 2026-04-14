"""
teleop_franka_hardware.py – Entry point for Meta Quest 3 → Franka FR3 teleoperation.

Usage
-----
    # Normal teleoperation:
    python scripts/hardware/teleop_franka_hardware.py

    # With Placo visualisation disabled (default) and custom robot IP:
    python scripts/hardware/teleop_franka_hardware.py --robot_ip 192.168.18.1

    # Dry-run (prints commands; does NOT move the robot):
    python scripts/hardware/teleop_franka_hardware.py --dry_run

    # Trigger homing immediately on startup (robot moves to home pose):
    python scripts/hardware/teleop_franka_hardware.py --reset

Controller Mapping (Meta Quest 3)
-----------------------------------
  Right grip (hold)   → Clutch ENGAGED  : robot follows hand motion
  Right grip (release)→ Clutch DISENGAGED: robot holds last position
  Right trigger       → Gripper close/open (analog, threshold 0.5)
  Left grip           → Start episode (triggers homing → teleop)
  Left trigger        → Stop episode  (saves data)

Data Collection
---------------
  /robot_state  (sensor_msgs/JointState) published at 10 Hz when active.
  /episode_control (std_msgs/String) "START"/"STOP" for episode_recorder.py.

  To record an episode, also start:
    ros2 run <cam_pkg> cam_pub.py         (for each camera)
    python scripts/dataset/episode_recorder.py

Network Requirements
--------------------
  Robot PC (franka_pose_cmd_client) must be running and reachable at --robot_ip.
    Command port  : --cmd_port  (default 8888)
    State port    : --state_port (default 9093)
  XRoboToolkit SDK server must be paired with the Meta Quest 3 headset.
"""

import threading
import time

import numpy as np
import tyro

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.franka_teleop_controller import (
    DEFAULT_MANIPULATOR_CONFIG,
    ArmFrankaIncController,
    WORKSPACE_BOUNDS,
)
from xrobotoolkit_teleop.hardware.interface.franka_robot import (
    DEFAULT_CMD_PORT_FULL,
    DEFAULT_FRANKA_IP,
    DEFAULT_STATE_PORT,
)
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD


def main(
    robot_ip: str = DEFAULT_FRANKA_IP,
    cmd_port: int = DEFAULT_CMD_PORT_FULL,
    state_port: int = DEFAULT_STATE_PORT,
    scale_factor: float = 1.0,
    reset: bool = False,
    dry_run: bool = False,
    publish_state_mode: str = "internal",
    home_x: float = 0.3,
    home_y: float = 0.0,
    home_z: float = 0.4,
) -> None:
    """
    Run Meta Quest 3 → Franka FR3 teleoperation.

    Args:
        robot_ip:           IP of the Franka robot PC (franka_pose_cmd_client).
        cmd_port:           UDP command port (8888 = pose+gripper, 8890 = pose only).
        state_port:         UDP state stream port.
        scale_factor:       Scale applied to XR controller motion (1.0 = 1:1).
        reset:              If True, trigger homing immediately on startup.
        dry_run:            If True, print commands instead of sending to robot.
        publish_state_mode: "internal" (publish /robot_state here) or
                            "external" (notify robot_pub via /collect_cmd).
        home_x/y/z:         Home EE position [m] in robot base frame.
    """
    print("=" * 60)
    print("  Franka FR3 XR Teleoperation")
    print("=" * 60)
    print(f"  Robot IP        : {robot_ip}")
    print(f"  Command port    : {cmd_port}")
    print(f"  State port      : {state_port}")
    print(f"  Scale factor    : {scale_factor}")
    print(f"  Dry run         : {dry_run}")
    print(f"  Home position   : [{home_x:.3f}, {home_y:.3f}, {home_z:.3f}]")
    print(f"  Workspace X     : {WORKSPACE_BOUNDS['x']}")
    print(f"  Workspace Y     : {WORKSPACE_BOUNDS['y']}")
    print(f"  Workspace Z     : {WORKSPACE_BOUNDS['z']}")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Initialise XRoboToolkit SDK and controller
    # ---------------------------------------------------------------
    xr_client = XrClient()

    home_pos = np.array([home_x, home_y, home_z])
    home_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity orientation

    arm_controller = ArmFrankaIncController(
        xr_client=xr_client,
        robot_ip=robot_ip,
        robot_cmd_port=cmd_port,
        robot_state_port=state_port,
        home_pos=home_pos,
        home_quat_xyzw=home_quat,
        R_headset_world=R_HEADSET_TO_WORLD,
        scale_factor=scale_factor,
        manipulator_config=DEFAULT_MANIPULATOR_CONFIG,
        publish_state_mode=publish_state_mode,
        dry_run=dry_run,
    )

    # ---------------------------------------------------------------
    # Optional: immediate homing / mode setup
    # ---------------------------------------------------------------
    if reset:
        print("[Main] --reset flag set: triggering homing on startup.")
        arm_controller.reset()
    else:
        # Start in teleop mode directly (no homing)
        arm_controller.set_mode("teleop")

    # ---------------------------------------------------------------
    # Launch threads
    # ---------------------------------------------------------------
    stop_signal = threading.Event()

    arm_control_thread = threading.Thread(
        target=arm_controller.run_ik_and_control_thread,
        args=(stop_signal,),
        name="franka-control",
        daemon=True,
    )
    gripper_control_thread = threading.Thread(
        target=arm_controller.run_gripper_control_thread,
        args=(stop_signal,),
        name="franka-gripper",
        daemon=True,
    )
    collect_state_thread = threading.Thread(
        target=arm_controller.run_collect_state_thread,
        args=(stop_signal,),
        name="franka-collect-state",
        daemon=True,
    )
    episode_control_thread = threading.Thread(
        target=arm_controller.run_episode_control_thread,
        args=(stop_signal,),
        name="franka-episode-ctrl",
        daemon=True,
    )

    arm_control_thread.start()
    gripper_control_thread.start()
    collect_state_thread.start()
    episode_control_thread.start()

    print("\n[Main] Teleoperation running.  Press Ctrl+C to exit.")
    print("[Main] Hold RIGHT GRIP     → clutch ON: robot follows your hand.")
    print("[Main] Release RIGHT GRIP  → clutch OFF: robot holds last position.")
    print("[Main] RIGHT TRIGGER       → gripper toggle (squeeze=close, release=open).")
    print("[Main] Press LEFT GRIP     → start a data episode.")
    print("[Main] Press LEFT TRIGGER  → stop & save a data episode.\n")

    try:
        while not stop_signal.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received – shutting down...")
        stop_signal.set()

    # Wait for all threads to finish
    for t in (arm_control_thread, gripper_control_thread, collect_state_thread, episode_control_thread):
        t.join(timeout=2.0)

    arm_controller.close()
    print("[Main] All threads stopped.  Goodbye.")


if __name__ == "__main__":
    tyro.cli(main)

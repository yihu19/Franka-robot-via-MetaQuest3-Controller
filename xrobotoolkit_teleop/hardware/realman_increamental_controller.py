import os
import time
import webbrowser
import sys
import threading

import meshcat.transformations as tf
import numpy as np
import placo
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.realman_robots import (
    CONTROLLER_DEADZONE,
    GRIPPER_FORCE,
    GRIPPER_SPEED,
    LOOKAHEAD_TIME,
    MAX_ACCELERATION,
    MAX_VELOCITY,
    # RIGHT_INITIAL_JOINT_DEG,
    # RIGHT_ROBOT_IP,
    RIGHT_INITIAL_JOINT_DEG,
    RIGHT_ROBOT_IP,
    SERVO_GAIN,
    SERVO_TIME,
    RealManController,
)
from xrobotoolkit_teleop.utils.ros2_command import Ros2CollectorClient
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import (
    calc_parallel_gripper_position,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


DEFAULT_DUAL_ARM_URDF_PATH = os.path.join(ASSET_PATH, "realman/rml_63b/RML-63-B-hardware.urdf")
DEFAULT_SCALE_FACTOR = 1.2
GRIPPPER_DEFAULT_SPEED = 0.5

DEFAULT_MANIPULATOR_CONFIG = {
    "right_arm": {
        "link_name": "Link_6",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_trigger": "right_trigger",

        "start_button": "left_grip",
        "stop_button":  "left_trigger",
        "threshold":    0.8,
        "debounce_s":   0.3,
    },
}

class ArmRealManIncController:
    def __init__(
        self,
        xr_client: XrClient,
        robot_urdf_path: str = DEFAULT_DUAL_ARM_URDF_PATH,  # Path to URDF for Placo
        manipulator_config: dict = DEFAULT_MANIPULATOR_CONFIG,
        right_initial_joint_deg: np.ndarray = RIGHT_INITIAL_JOINT_DEG,  # Use DEG for consistency
        max_velocity: float = MAX_VELOCITY,
        max_acceleration: float = MAX_ACCELERATION,
        servo_time: float = SERVO_TIME,
        lookahead_time: float = LOOKAHEAD_TIME,
        servo_gain: float = SERVO_GAIN,
        gripper_force: float = GRIPPER_FORCE,
        gripper_speed: float = GRIPPER_SPEED,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = True,  # Add placo visualization option
        collect_change_threshold: float = 0.01,   # Trigger change threshold (0~1)
        collect_idle_timeout: float = 0.5,        # Send STOP after idle timeout (seconds)
        publish_state_mode: str = "internal",  # "internal" | "external"
    ):
        self.xr_client = xr_client
        self.robot_urdf_path = robot_urdf_path
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.visualize_placo = visualize_placo

        self.collect_change_threshold = collect_change_threshold
        self.collect_idle_timeout = collect_idle_timeout

        # New: state publishing mode
        self.publish_state_mode = (publish_state_mode or "internal").lower()
        if self.publish_state_mode not in ("internal", "external"):
            print(f"[Collector] unknown publish_state_mode={self.publish_state_mode}, fallback to 'internal'")
            self.publish_state_mode = "internal"

        # Data collection component
        self.collector_client = Ros2CollectorClient(mode=self.publish_state_mode)
        if self.publish_state_mode == "internal":
            # Register provider to collector client; automatically publish to /robot_state at 10Hz after start_collect
            self.collector_client.set_state_provider(
                self._state_provider,
                topic="/robot_state",
                rate_hz=10.0,
                reliable=True,
            )

        self._ep_prev_start = 0
        self._ep_prev_stop  = 0
        self._ep_last_ns    = 0
        self.episode_collecting = False
        # Episode/IK mode: idle | homing | teleop
        self.mode = "idle"
        # Homing target and tolerance (in degrees)
        self.right_home_q_deg = np.asarray(right_initial_joint_deg, dtype=float).reshape(-1)
        self.home_tol_deg = 1.0
        self._homing_started = False

        self.right_controller = RealManController(
            initial_joint_positions=right_initial_joint_deg,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            servo_time=servo_time,
            lookahead_time=lookahead_time,
            servo_gain=servo_gain,
            gripper_force=gripper_force,
            gripper_speed=gripper_speed,
            wifi=False,
        )

        self.gripper_active = False

        # Placo Setup
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = servo_time
        self.solver.mask_fbase(True)
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Define end-effector configuration (adjust link names and pico sources as needed)
        self.manipulator_config = manipulator_config

        self.effector_task = {}
        self.init_controller_xyz = {}
        self.init_controller_quat = {}


        # New: persistent target end-effector pose for each arm (not reset with actual end-effector)
        self.ee_target_T = {}

        # New: data collection state tracking
        self.prev_gripper_val = {}
        self.collecting = {}
        self.last_activity_ts = {}
        self.trigger_val = {}

        for name, config in self.manipulator_config.items():
            initial_pose = np.eye(4)
            self.effector_task[name] = self.solver.add_frame_task(config["link_name"], initial_pose)
            self.effector_task[name].configure(f"{name}_frame", "soft", 1.0)
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability.configure(f"{name}_manipulability", "soft", 5e-2)

            self.init_controller_xyz[name] = np.array([0, 0, 0])
            self.init_controller_quat[name] = np.array([1, 0, 0, 0])

            self.ee_target_T[name] = np.eye(4)

            # New: initialize data collection state
            self.prev_gripper_val[name] = 0.0
            self.collecting[name] = False
            self.last_activity_ts[name] = 0.0

        right_q_init = self.right_controller.get_current_joint_positions()

        # self.left_gripper_pos = 1000
        self.right_gripper_pos = self.right_controller.get_gripper_open_position()

        # 7 (base) + 6 (right)
        self.placo_robot.state.q[7:13] = right_q_init
        self.placo_robot.update_kinematics()

        self.target_right_q = right_q_init.copy()

        if self.visualize_placo:
            self.placo_robot.update_kinematics()
            self.placo_vis = robot_viz(self.placo_robot)

            # Automatically open browser window
            time.sleep(0.5)  # Small delay to ensure server is ready
            meshcat_url = self.placo_vis.viewer.url()
            print(f"Automatically opening meshcat at: {meshcat_url}")
            webbrowser.open(meshcat_url)

            self.placo_vis.display(self.placo_robot.state.q)
            for name, config in self.manipulator_config.items():
                robot_frame_viz(self.placo_robot, config["link_name"])
                frame_viz(
                    f"vis_target_{name}",
                    self.effector_task[name].T_world_frame,
                )

        self._js_prev_q = None
        self._js_prev_ns = None

    def _state_provider(self):
        # Read joint and end-effector state
        q, ee_pose = self.right_controller.getState()
        # Convert to radians, ensure as list[float]
        try:
            q = (np.asarray(q, dtype=float) * np.pi / 180.0).reshape(-1).tolist()
        except Exception:
            q = [float(q)] if q is not None else []
        if len(q) < 6:
            q = q + [0.0] * (6 - len(q))
        else:
            q = q[:6]

        try:
            ee_pose = np.asarray(ee_pose, dtype=float).reshape(-1).tolist()
        except Exception:
            ee_pose = [float(ee_pose)] if ee_pose is not None else []
        if len(ee_pose) < 6:
            ee_pose = ee_pose + [0.0] * (6 - len(ee_pose))
        else:
            ee_pose = ee_pose[:6]

        now_ns = time.time_ns()

        # Calculate joint velocity
        if self._js_prev_q is not None and self._js_prev_ns is not None and now_ns > self._js_prev_ns:
            dt = (now_ns - self._js_prev_ns) / 1e9
            qdot = [0.0] * 6 if dt < 1e-6 else [(q[i] - self._js_prev_q[i]) / dt for i in range(6)]
        else:
            qdot = [0.0] * 6

        # Get gripper position/speed (note: interface returns two floats)
        gripper_pos, gripper_speed = self.right_controller.get_gripper_state()
        gripper_pos = max(0.0, min(1.0, float(gripper_pos)))
        gripper_speed = float(gripper_speed)

        # Combine with field order consistent with subscriber
        names = [
            "joint1","joint2","joint3","joint4","joint5","joint6",
            "gripper",
            "ee_x","ee_y","ee_z","ee_rx","ee_ry","ee_rz",
            "joint1_speed","joint2_speed","joint3_speed","joint4_speed","joint5_speed","joint6_speed",
            "gripper_speed",
        ]
        positions = list(q) + [gripper_pos] + list(ee_pose) + qdot + [gripper_speed]

        # Update history
        self._js_prev_q = list(q)
        self._js_prev_ns = now_ns

        return {
            "name": names,
            "position": positions,
            "frame_id": "base_link",
        }
    def _process_xr_pose(self, xr_pose, arm_name: str, R_world_ee: np.ndarray):
        """Returns increment relative to current arm end-effector frame (EE local frame): (delta_xyz_local, delta_rotvec_local)"""
        # xr_pose is typically [tx, ty, tz, qx, qy, qz, qw]
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = np.array([xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]])  # w, x, y, z

        # Transform controller pose to world frame
        controller_xyz = self.R_headset_world @ controller_xyz
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )

        if self.init_controller_xyz[arm_name] is None:
            self.init_controller_xyz[arm_name] = controller_xyz.copy()
            self.init_controller_quat[arm_name] = controller_quat.copy()
            return np.zeros(3), np.zeros(3)

        # Increment relative to previous frame in world frame
        delta_xyz_world = (controller_xyz - self.init_controller_xyz[arm_name]) * self.scale_factor
        delta_rot_world = quat_diff_as_angle_axis(self.init_controller_quat[arm_name], controller_quat)

        # Update previous frame cache
        self.init_controller_xyz[arm_name] = controller_xyz.copy()
        self.init_controller_quat[arm_name] = controller_quat.copy()

        # World frame -> current actual end-effector local frame
        R_ee_world = R_world_ee.T
        delta_xyz_local = R_ee_world @ delta_xyz_world
        delta_rot_local = R_ee_world @ delta_rot_world
        return delta_xyz_local, delta_rot_local

    def calc_target_joint_position(self):
        """
        Calculates the target joint positions for both arms using Placo IK
        based on Pico controller poses and grip commands.
        """
        # 1) If not recording or in idle mode, don't issue IK commands
        if self.mode == "idle":
            return

        # 2) Homing phase: let the low-level controller home, IK does not intervene until homed
        if self.mode == "homing":
            # Trigger reset on first entry to homing
            if not self._homing_started:
                try:
                    print("[Episode] Homing: resetting arm to initial joints...", flush=True)
                    self.right_controller.reset()
                except Exception as e:
                    print(f"[Episode] reset() failed: {e}", flush=True)
                self._homing_started = True
            try:
                curr = np.asarray(self.right_controller.get_current_joint_degrees(), dtype=float).reshape(-1)
                err = float(np.max(np.abs(curr - self.right_home_q_deg)))
            except Exception:
                err = self.home_tol_deg + 1.0
            if err > self.home_tol_deg:
                # Not yet homed, wait for low-level motion to complete
                return
            # Homed -> switch to teleop, reset cache to avoid first-frame jump
            for arm_name in self.manipulator_config.keys():
                self.ee_target_T[arm_name] = np.eye(4)
                self.init_controller_xyz[arm_name] = np.zeros(3)
                self.init_controller_quat[arm_name] = np.zeros(4)
            self.mode = "teleop"
            print("[Episode] Homing complete. Teleop enabled.")
            return

        # current_q_left_actual = self.static_left_q
        current_q_right_actual = self.right_controller.get_current_joint_positions()

        # self.placo_robot.state.q[7:13] = current_q_left_actual
        self.placo_robot.state.q[7:13] = current_q_right_actual

        self.placo_robot.update_kinematics()

        for arm_name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)
            self.gripper_active = active

            T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
            R_world_ee_cur = T_world_ee_current[:3, :3]

            if active:
                if self.ee_target_T[arm_name] is None:
                    self.ee_target_T[arm_name] = T_world_ee_current.copy()
                    # Clear previous frame controller cache
                    self.init_controller_xyz[arm_name] = None
                    self.init_controller_quat[arm_name] = None

                    p_world_ee_cur = T_world_ee_current[:3, 3]
                    p_world_ee_quat_cur = tf.quaternion_from_matrix(T_world_ee_current)
                    print(f"{arm_name} activated. Current EE xyz: {p_world_ee_cur}, quat: {p_world_ee_quat_cur}.")

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz_local_cur, delta_rot_local_cur = self._process_xr_pose(xr_pose, arm_name, R_world_ee_cur)

                # Convert increment (defined in current actual end-effector frame) to target end-effector frame local representation
                R_world_ee_des = self.ee_target_T[arm_name][:3, :3]
                R_des_cur = R_world_ee_des.T @ R_world_ee_cur
                delta_xyz_local_des = R_des_cur @ delta_xyz_local_cur
                delta_rot_local_des = R_des_cur @ delta_rot_local_cur

                # Right-multiply to persistent target: T_des = T_des @ Exp(Δ_local_des)
                theta = np.linalg.norm(delta_rot_local_des)
                if theta > 1e-9:
                    axis = delta_rot_local_des / theta
                    q_inc = tf.quaternion_about_axis(theta, axis)  # [w,x,y,z]
                    R_inc = tf.quaternion_matrix(q_inc)[:3, :3]
                else:
                    R_inc = np.eye(3)

                T_inc = np.eye(4)
                T_inc[:3, :3] = R_inc
                T_inc[:3, 3] = delta_xyz_local_des

                self.ee_target_T[arm_name] = self.ee_target_T[arm_name] @ T_inc
                self.effector_task[arm_name].T_world_frame = self.ee_target_T[arm_name]
                
            else:
                if self.ee_target_T[arm_name] is not None:
                    print(f"{arm_name} deactivated.")
                    # Deactivation: target returns to current, clear cache
                    self.ee_target_T[arm_name] = None
                    self.init_controller_xyz[arm_name] = None
                    self.init_controller_quat[arm_name] = None
                    self.effector_task[arm_name].T_world_frame = T_world_ee_current

        try:
            self.solver.solve(True)

            # self.target_left_q = self.placo_robot.state.q[7:13].copy()
            self.target_right_q = self.placo_robot.state.q[7:13].copy()
            # print(f"Target right q (deg): {self.target_right_q * 180.0 / np.pi}")

            self.right_controller.servo_joints(self.target_right_q)

            if self.visualize_placo and hasattr(self, "placo_vis"):
                self.placo_vis.display(self.placo_robot.state.q)
                for name, config in self.manipulator_config.items():
                    robot_frame_viz(self.placo_robot, config["link_name"])
                    frame_viz(
                        f"vis_target_{name}",
                        self.effector_task[name].T_world_frame,
                    )

        except RuntimeError as e:
            print(f"IK solver failed: {e}. Returning last known good joint positions.")
        except Exception as e:
            print(f"An unexpected error occurred in IK: {e}. Returning last known good joint positions.")

    def control_gripper(self):
        if self.mode != "teleop":
            return
        
        for arm_name, config in self.manipulator_config.items():
            trigger_val = self.xr_client.get_key_value_by_name(config["gripper_trigger"])
            self.trigger_val[arm_name] = trigger_val
            if arm_name == "right_arm":
                self.right_gripper_pos = int(
                    calc_parallel_gripper_position(
                        self.right_controller.get_gripper_open_position(),
                        self.right_controller.get_gripper_close_position(),
                        trigger_val,
                    )
                )

        self.right_controller.set_gripper_position(
                self.right_gripper_pos
                # self.right_controller.gripper_speed,
                # self.right_controller.gripper_force,
            )
        
    def update_collect_state(self):
        # Based on mode: internal -> publish /robot_state in this process; external -> notify robot_pub to publish via /collect_cmd
        for arm_name, _ in self.manipulator_config.items():
            prev = self.prev_gripper_val[arm_name]
            cur = self.trigger_val.get(arm_name, 0.0)
            changed = (prev is None) or (abs(cur - prev) > self.collect_change_threshold)
            self.prev_gripper_val[arm_name] = cur

            now = time.time()
            if self.gripper_active or changed:
                self.last_activity_ts[arm_name] = now
                if not self.collecting[arm_name]:
                    self.collecting[arm_name] = True
                    try:
                        if self.publish_state_mode == "internal":
                            self.collector_client.start_collect()       # Enable this process /robot_state periodic publishing
                        else:
                            self.collector_client.start_collect_cmd()   # Notify robot_pub to start publishing
                    except Exception as e:
                        print(f"[Collector] start ({self.publish_state_mode}) error: {e}")
            else:
                if self.collecting[arm_name] and (now - self.last_activity_ts[arm_name] >= self.collect_idle_timeout):
                    self.collecting[arm_name] = False
                    try:
                        if self.publish_state_mode == "internal":
                            self.collector_client.stop_collect()
                        else:
                            self.collector_client.stop_collect_cmd()
                    except Exception as e:
                        print(f"[Collector] stop ({self.publish_state_mode}) error: {e}")

    def _publish_episode_cmd_async(self, cmd: str):
        def _worker():
            try:
                if cmd == "START":
                    self.collector_client.start_episode()
                elif cmd == "STOP":
                    self.collector_client.stop_episode()
            except Exception as e:
                print(f"[Episode] publish '{cmd}' error: {e}", flush=True)
        threading.Thread(target=_worker, daemon=True).start()

    def _edge_pressed(self, cur: float, prev: int, th: float) -> bool:
        return (cur >= th) and (prev == 0)

    def update_episode_control(self):
        # Control data collection start and stop via specified triggers
        v_start = self.xr_client.get_key_value_by_name(
            self.manipulator_config["right_arm"]["start_button"]
        )
        v_stop = self.xr_client.get_key_value_by_name(
            self.manipulator_config["right_arm"]["stop_button"]
        )

        now_ns = time.time_ns()
        debounce_ok = (now_ns - self._ep_last_ns) >= int(self.manipulator_config["right_arm"]["debounce_s"] * 1e9)

        if debounce_ok and (not self.episode_collecting) and self._edge_pressed(v_start, self._ep_prev_start, self.manipulator_config["right_arm"]["threshold"]):
            # Start episode
            self.episode_collecting = True
            self._ep_last_ns = now_ns
            # Publish asynchronously to avoid blocking this thread
            # Enter homing mode; reset is triggered on first iteration in IK thread
            self.mode = "homing"
            self._homing_started = False
            self._publish_episode_cmd_async("START")


        if debounce_ok and self.episode_collecting and self._edge_pressed(v_stop, self._ep_prev_stop, self.manipulator_config['right_arm']['threshold']):
            # Stop episode
            self.episode_collecting = False
            self._ep_last_ns = now_ns
            self._publish_episode_cmd_async("STOP")
            print("[Episode] Stop data collection.", flush=True)
            # New: enter idle mode immediately to avoid continuing control commands
            self.mode = "teleop"
            self._homing_started = False

        self._ep_prev_start = 1 if v_start >= self.manipulator_config["right_arm"]["threshold"] else 0
        self._ep_prev_stop  = 1 if v_stop  >= self.manipulator_config["right_arm"]["threshold"] else 0

    def run_gripper_control_thread(self, stop_event):
        print("Starting right gripper control thread...")
        while not stop_event.is_set():
            self.control_gripper()
            time.sleep(0.001)

    def run_ik_and_control_thread(self, stop_event):
        while not stop_event.is_set():
            try:
                # start = time.time()
                self.calc_target_joint_position()
                # elapsed = time.time() - start
                # time.sleep(max(0, 0.01 - elapsed))
            except Exception as e:
                print(f"Error in IK calculation: {e}")

        self.right_controller.close()

    def run_collect_state_thread(self, stop_event):
        print("Starting state collection thread...")
        while not stop_event.is_set():
            self.update_collect_state()
            time.sleep(0.5)  # 2 Hz   

    def run_episode_control_thread(self, stop_event):
        print("Starting episode control thread...")
        while not stop_event.is_set():
            self.update_episode_control()
            time.sleep(0.01)  # 100 Hz

    def reset(self):
        self.right_controller.reset()

    def close(self):
        # self.collector_client.close()
        # self.right_controller.close()
        try:
            self.collector_client.close()
        except Exception:
            pass
        try:
            self.right_controller.close()
        except Exception:
            pass

    def __del__(self):
        """Ensures resources are released when the object is deleted."""
        self.close()

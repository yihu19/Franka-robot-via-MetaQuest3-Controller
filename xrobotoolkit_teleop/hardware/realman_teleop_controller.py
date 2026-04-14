import os
import threading
import time
import webbrowser

import meshcat.transformations as tf
import numpy as np
import placo
import threading
import matplotlib.pyplot as plt
import queue

from collections import deque
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
    SERVO_GAIN,
    SERVO_TIME,
    RealManController,
)
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import (
    calc_parallel_gripper_position,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

DEFAULT_DUAL_ARM_URDF_PATH = os.path.join(ASSET_PATH, "realman_63b/RML-63-B-gripper_hardware.urdf")
DEFAULT_SCALE_FACTOR = 1.0

DEFAULT_MANIPULATOR_CONFIG = {
    # "left_arm": {
    #     "link_name": "left_tool0",
    #     "pose_source": "left_controller",
    #     "control_trigger": "left_grip",
    #     "gripper_trigger": "left_trigger",
    # },
    "right_arm": {
        "link_name": "Link_6",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_trigger": "right_trigger",
    },
}

class ArmRealManAbsController:
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
        visualize_placo: bool = False,  
        plot_controller_data: bool = True,  
        history_length: int = 300,          
    ):
        self.xr_client = xr_client
        self.robot_urdf_path = robot_urdf_path
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor

        self.visualize_placo = visualize_placo
        self.plot_controller_data = plot_controller_data
        self.history_length = history_length

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

        # Placo Setup
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = servo_time
        self.solver.mask_fbase(True)
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Define end-effector configuration (adjust link names and pico sources as needed)
        self.manipulator_config = manipulator_config

        self.effector_task = {}
        self.init_ee_xyz = {}
        self.init_ee_quat = {}
        self.init_controller_xyz = {}
        self.init_controller_quat = {}
        for name, config in self.manipulator_config.items():
            initial_pose = np.eye(4)
            self.effector_task[name] = self.solver.add_frame_task(config["link_name"], initial_pose)
            self.effector_task[name].configure(f"{name}_frame", "soft", 1.0)
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability.configure(f"{name}_manipulability", "soft", 5e-2)
            self.init_ee_xyz[name] = np.array([0, 0, 0])
            self.init_ee_quat[name] = np.array([1, 0, 0, 0])
            self.init_controller_xyz[name] = np.array([0, 0, 0])
            self.init_controller_quat[name] = np.array([1, 0, 0, 0])

        # left_q_init = np.deg2rad(right_initial_joint_deg)
        # self.static_left_q = left_q_init.copy()
        right_q_init = self.right_controller.get_current_joint_positions()
        # right_pose_init = self.right_controller.get_current_tcp_pose()

        # self.left_gripper_pos = 1000
        self.right_gripper_pos = self.right_controller.get_gripper_open_position()

        # 7 (base) + 6 (left) + 6 (right)
        # self.placo_robot.state.q[7:13] = self.static_left_q
        self.placo_robot.state.q[7:13] = right_q_init

        # self.target_left_q = self.static_left_q.copy()
        self.target_right_q = right_q_init.copy()
        # self.target_right_pose = right_pose_init.copy()

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

        # Initialize data structures only when plotting is needed
        if self.plot_controller_data:
            # Initialize data storage
            self.timestamp = deque(maxlen=history_length)
            
            # Controller position data
            self.controller_pos_x = deque(maxlen=history_length)
            self.controller_pos_y = deque(maxlen=history_length)
            self.controller_pos_z = deque(maxlen=history_length)
            
            # Controller pose data (Euler angles)
            self.controller_roll = deque(maxlen=history_length)
            self.controller_pitch = deque(maxlen=history_length)
            self.controller_yaw = deque(maxlen=history_length)
            
            # Set start time
            self.start_time = time.time()

            # Create data queue instead of starting thread
            self.plot_data_queue = queue.Queue()
            
            # Setup plotting window
            self._setup_plot()

    def _process_xr_pose(self, xr_pose, arm_name: str):
        """Process the current XR controller pose, similar to MujocoTeleopController."""
        # xr_pose is typically [tx, ty, tz, qx, qy, qz, qw]
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = np.array(
            [
                xr_pose[6],  # w
                xr_pose[3],  # x
                xr_pose[4],  # y
                xr_pose[5],  # z
            ]
        )

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
            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])  # Angle-axis
        else:
            delta_xyz = (controller_xyz - self.init_controller_xyz[arm_name]) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.init_controller_quat[arm_name], controller_quat)
        return delta_xyz, delta_rot
    
    
    def calc_target_joint_position(self):
        """
        Calculates the target joint positions for both arms using Placo IK
        based on Pico controller poses and grip commands.
        """
        # current_q_left_actual = self.static_left_q
        current_q_right_actual = self.right_controller.get_current_joint_positions()

        # self.placo_robot.state.q[7:13] = current_q_left_actual
        self.placo_robot.state.q[7:13] = current_q_right_actual

        self.placo_robot.update_kinematics()

        for arm_name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)
            trigger_val = self.xr_client.get_key_value_by_name(config["gripper_trigger"])
            if arm_name == "right_arm":
                self.right_gripper_pos = int(
                    calc_parallel_gripper_position(
                        self.right_controller.get_gripper_open_position(),
                        self.right_controller.get_gripper_close_position(),
                        trigger_val,
                    )
                )

            if active:
                if self.init_ee_xyz[arm_name] is None:
                    # First activation: store current EE pose as initial
                    # Get current EE pose from Placo model based on actual joint angles
                    T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
                    self.init_ee_xyz[arm_name] = T_world_ee_current[:3, 3].copy()
                    self.init_ee_quat[arm_name] = tf.quaternion_from_matrix(T_world_ee_current)
                    print(
                        f"{arm_name} activated. Current EE xyz: {self.init_ee_xyz[arm_name]}, quat: {self.init_ee_quat[arm_name]}."
                    )

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot_angle_axis = self._process_xr_pose(xr_pose, arm_name)

                target_xyz, target_quat = apply_delta_pose(
                    self.init_ee_xyz[arm_name],
                    self.init_ee_quat[arm_name],
                    delta_xyz,
                    delta_rot_angle_axis,
                )

                target_transform = tf.quaternion_matrix(target_quat)
                target_transform[:3, 3] = target_xyz
                self.effector_task[arm_name].T_world_frame = target_transform

                # self.target_right_pose[:3] = target_xyz
                # self.target_right_pose[3:] = self.target_right_pose[3:] + delta_rot_angle_axis

            else:  # Not active
                if self.init_ee_xyz[arm_name] is not None:
                    print(f"{arm_name} deactivated.")
                    self.init_ee_xyz[arm_name] = None
                    self.init_ee_quat[arm_name] = None
                    self.init_controller_xyz[arm_name] = None
                    self.init_controller_quat[arm_name] = None
                    T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
                    self.effector_task[arm_name].T_world_frame = T_world_ee_current

                    # self.target_right_pose = self.right_controller.get_current_tcp_pose()

        try:
            self.solver.solve(True)

            # self.target_left_q = self.placo_robot.state.q[7:13].copy()
            self.target_right_q = self.placo_robot.state.q[7:13].copy()

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


    def _setup_plot(self):
        """Initialize the controller data plotting window"""

        plt.ion()  # Enable interactive mode
        self.fig, self.axs = plt.subplots(2, 3, figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('VR Controller Pose Data')
        
        # Position plots
        self.axs[0, 0].set_title('Pose X')
        self.axs[0, 0].set_xlabel('Time (s)')
        self.axs[0, 0].set_ylabel('Position (m)')
        self.axs[0, 0].grid(True)
        self.line_x, = self.axs[0, 0].plot([], [], 'r-')
        
        self.axs[0, 1].set_title('Pose Y')
        self.axs[0, 1].set_xlabel('Time (s)')
        self.axs[0, 1].set_ylabel('Position (m)')
        self.axs[0, 1].grid(True)
        self.line_y, = self.axs[0, 1].plot([], [], 'g-')
        
        self.axs[0, 2].set_title('Pose Z')
        self.axs[0, 2].set_xlabel('Time (s)')
        self.axs[0, 2].set_ylabel('Position (m)')
        self.axs[0, 2].grid(True)
        self.line_z, = self.axs[0, 2].plot([], [], 'b-')
        
        # Attitude plots
        self.axs[1, 0].set_title('Roll')
        self.axs[1, 0].set_xlabel('Time (s)')
        self.axs[1, 0].set_ylabel('Angle (degrees)')
        self.axs[1, 0].grid(True)
        self.line_roll, = self.axs[1, 0].plot([], [], 'r-')
        
        self.axs[1, 1].set_title('Pitch')
        self.axs[1, 1].set_xlabel('Time (s)')
        self.axs[1, 1].set_ylabel('Angle (degrees)')
        self.axs[1, 1].grid(True)
        self.line_pitch, = self.axs[1, 1].plot([], [], 'g-')
        
        self.axs[1, 2].set_title('Yaw')
        self.axs[1, 2].set_xlabel('Time (s)')
        self.axs[1, 2].set_ylabel('Angle (degrees)')
        self.axs[1, 2].grid(True)
        self.line_yaw, = self.axs[1, 2].plot([], [], 'b-')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.01)

    def update_plots(self):
        """Call this method in the main thread to update the plots"""
        if not self.plot_controller_data:
            return
            
        # Retrieve all available data from the queue
        updated = False
        while not self.plot_data_queue.empty():
            try:
                data = self.plot_data_queue.get_nowait()
                self.timestamp.append(data['time'])
                self.controller_pos_x.append(data['pos_x'])
                self.controller_pos_y.append(data['pos_y'])
                self.controller_pos_z.append(data['pos_z'])
                self.controller_roll.append(data['roll'])
                self.controller_pitch.append(data['pitch'])
                self.controller_yaw.append(data['yaw'])
                updated = True
            except queue.Empty:
                break
        
        # Update plots only when there is new data
        if updated and len(self.timestamp) >= 2:
            self.line_x.set_data(list(self.timestamp), list(self.controller_pos_x))
            self.line_y.set_data(list(self.timestamp), list(self.controller_pos_y))
            self.line_z.set_data(list(self.timestamp), list(self.controller_pos_z))
            self.line_roll.set_data(list(self.timestamp), list(self.controller_roll))
            self.line_pitch.set_data(list(self.timestamp), list(self.controller_pitch))
            self.line_yaw.set_data(list(self.timestamp), list(self.controller_yaw))
            
            for i in range(2):
                for j in range(3):
                    self.axs[i, j].relim()
                    self.axs[i, j].autoscale_view()
            
            self.fig.canvas.draw_idle()
            plt.pause(0.001)  # A very short pause to allow GUI event handling

    def run_plot_controller_pose_thread(self, stop_event):
        """Thread function: continuously collect controller pose data and put into queue"""
        print("Starting controller pose data collection thread...")
        
        if not self.plot_controller_data:
            print("Plot controller data is disabled. Thread exiting.")
            return
        
        while not stop_event.is_set():
            try:
                for arm_name, config in self.manipulator_config.items():
                    # Get raw controller data
                    xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                    controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
                    controller_quat = np.array([xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]])
                    
                    # Transform to world coordinate system
                    controller_xyz = self.R_headset_world @ controller_xyz
                    
                    # Transform rotation
                    R_transform = np.eye(4)
                    R_transform[:3, :3] = self.R_headset_world
                    R_quat = tf.quaternion_from_matrix(R_transform)
                    controller_quat = tf.quaternion_multiply(
                        tf.quaternion_multiply(R_quat, controller_quat),
                        tf.quaternion_conjugate(R_quat),
                    )
                    
                    # Convert to Euler angles (degrees)
                    controller_euler = np.degrees(tf.euler_from_quaternion(controller_quat))
                    
                    # Put data into queue
                    plot_data = {
                        'time': time.time() - self.start_time,
                        'pos_x': controller_xyz[0],
                        'pos_y': controller_xyz[1],
                        'pos_z': controller_xyz[2],
                        'roll': controller_euler[0],
                        'pitch': controller_euler[1],
                        'yaw': controller_euler[2]
                    }
                    
                    try:
                        self.plot_data_queue.put_nowait(plot_data)
                    except queue.Full:
                        # If queue is full, remove oldest data
                        try:
                            self.plot_data_queue.get_nowait()
                            self.plot_data_queue.put_nowait(plot_data)
                        except (queue.Empty, queue.Full):
                            pass
                
                # Control data collection frequency
                time.sleep(0.02)  # 50Hz data collection rate
                
            except Exception as e:
                print(f"Error in plot controller pose thread: {e}")
                time.sleep(0.1)  # Brief wait on error

    def run_right_controller_thread(self, stop_event):
        print("Starting right arm control thread...")
        while not stop_event.is_set():
            self.right_controller.servo_joints(self.target_right_q)
            # self.right_controller.servo_pose(self.target_right_pose)
            self.right_controller.set_gripper_position(
                self.right_gripper_pos
                # self.right_controller.gripper_speed,
                # self.right_controller.gripper_force,
            )
        self.right_controller.close()

    def run_ik_thread(self, stop_event):
        print("Starting IK calculation thread...")
        while not stop_event.is_set():
            try:
                start = time.time()
                self.calc_target_joint_position()
                elapsed = time.time() - start
                # print(f"IK calculation time: {elapsed:.8f} seconds")
                time.sleep(max(0, 0.01 - elapsed))
            except Exception as e:
                print(f"Error in IK calculation: {e}")

    def reset(self):
        self.right_controller.reset()

    def close(self):
        self.right_controller.close()

        # If plotting is enabled, add a flag to indicate data collection has stopped
        if self.plot_controller_data:
            # Stop data collection but keep the window open
            self.plot_controller_data = False  # This will prevent update_plots() method from updating the plots
  
    def run(self, stop_event=threading.Event()):
        try:
            self.reset()
            self.calc_target_joint_position()

        except Exception as e:
            print(f"Error moving to initial positions or setting up Placo initial state: {e}")
            self.close()
            return

        print("Starting dual-arm control loop...")

        try:
            # left_thread = threading.Thread(
            #     target=self.run_left_controller_thread,
            #     args=(stop_event,),
            # )
            right_thread = threading.Thread(
                target=self.run_right_controller_thread,
                args=(stop_event,),
            )
            # left_thread.start()
            right_thread.start()

            while not stop_event.is_set():
                try:
                    self.calc_target_joint_position()
                except KeyboardInterrupt:
                    print("Keyboard interrupt received. Stopping control loop.")
                    stop_event.set()

            # left_thread.join()
            right_thread.join()
        except Exception as e:
            print(f"Exception during control loop: {e}")

        self.close()

    def __del__(self):
        """Ensures resources are released when the object is deleted."""
        self.close()

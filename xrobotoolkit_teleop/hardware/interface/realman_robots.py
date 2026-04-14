import ctypes
import logging
import time
import numpy as np

from .rm_robot_interface import *

logging.basicConfig(level=logging.INFO)
logger_ = logging.getLogger(__name__)


Joints = ctypes.c_float * 6

LEFT_ROBOT_IP = "192.168.50.55"
RIGHT_ROBOT_IP = "192.168.50.195"

SERVO_TIME = 0.017
LOOKAHEAD_TIME = 0.1
SERVO_GAIN = 300.0
MAX_VELOCITY = 0.5
MAX_ACCELERATION = 1.0

GRIPPER_FORCE = 500
GRIPPER_SPEED = 0.5
CONTROLLER_DEADZONE = 0.1

# LEFT_INITIAL_JOINT_DEG = np.array([165.26, -47.50, 118.93, -38.96, 87.51, 149.56])
RIGHT_INITIAL_JOINT_DEG = np.array([-35.407, 5.368, -116.107, -4.17, -34.731, 5.578])

class RealManController():
    def __init__(
        self,
        initial_joint_positions: np.ndarray,
        max_velocity: float = MAX_VELOCITY,
        max_acceleration: float = MAX_ACCELERATION,
        servo_time: float = SERVO_TIME,
        lookahead_time: float = LOOKAHEAD_TIME,
        servo_gain: float = SERVO_GAIN,
        gripper_force: float = GRIPPER_FORCE,
        gripper_speed: float = GRIPPER_SPEED,
        wifi: bool = True,
    ):
        # self.robot_ip = robot_ip
        self.initial_joint_positions = initial_joint_positions
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.servo_time = servo_time
        self.lookahead_time = lookahead_time
        self.servo_gain = servo_gain
        self.gripper_force = gripper_force
        self.gripper_speed = gripper_speed

        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

        if not wifi:
            self.handle = self.arm.rm_create_robot_arm("192.168.1.18", 8080)
        else:
            self.handle = self.arm.rm_create_robot_arm("192.168.33.80", 8080)

        self.joints = []
        # self.pose = Pose()

    def reset(self):
        self.setGripperRelease()
        print("Gripper Open.")
        self.setJoints(self.initial_joint_positions, v=30, r=0, connect=0, block=1)
        print(f"Moving to initial joint positions: {self.initial_joint_positions}")
        print("Reached initial position.")

    def servo_joints(self, joint_positions: np.ndarray):
        # cur_joins = np.array(self.getState()[0]) 
        joint_positions = joint_positions / np.pi * 180.0
        self.setJoint_CANFD(joint_positions, follow=False)
        time.sleep(0.004)

    # def servo_joints(self, joint_positions: np.ndarray):
    #     joint_positions = joint_positions / np.pi * 180.0
    #     self.setJoints(joint_positions, r=0, connect=0, block=1)

    def servo_pose(self, target_pose: np.ndarray):
        self.arm.rm_movep_canfd(target_pose, follow=False)
        time.sleep(0.004)

    # def servo_pose(self, target_pose: np.ndarray):
    #     self.arm.rm_movep_follow(target_pose)
    #     time.sleep(0.001)

    def get_current_joint_positions(self) -> np.ndarray:
        return np.array(self.getState()[0]) * np.pi / 180.0
    
    def get_current_joint_degrees(self) -> np.ndarray:
        return np.array(self.getState()[0])
    
    def get_current_tcp_pose(self) -> np.ndarray:
        return np.array(self.getState()[1])
    
    def set_gripper_position(self, position, block=1, timeout=30):
        self.arm.rm_set_gripper_position(position, block, timeout)  # position: 0-1000
    
    def get_gripper_open_position(self):
        return 1000
    
    def get_gripper_close_position(self):
        return 1
        
    def close(self):
        self.arm.rm_delete_robot_arm()
        # logger_.info(f'Close socket connection with robotic arm')

    def __del__(self):
        self.arm.rm_delete_robot_arm()
        logger_.info(f'Close socket connection with robotic arm')

    def algo_set_joint_min_limit(self, joint_limit = [-178.0, -123.0, -178.0, -178.0, -178.0, -360.0]):
        self.arm.rm_algo_set_joint_min_limit(joint_limit = joint_limit)

    def getState(self):
        res = self.arm.rm_get_current_arm_state()
        # self.joints = [_ for _ in res[1]['joint']]
        joints = [res[1]['joint'][i] for i in range(6)]
        pose = [res[1]['pose'][i]    for i in range(6)]
        # print((joints, pose))
        return joints, pose

    def setJoints(self, joints, v = 30, **kwargs):
        self.arm.rm_movej(joints, v = v, **kwargs)

    def setGripperRelease(self, speed=1000, block=True, timeout=3):
        self.arm.rm_set_gripper_release(speed, block, timeout)  # position: 0-1000

    def setJoints_CANFD(self, joints, follow):
        # Pass-through bypasses the robotic arm's algorithm for direct joint motion control.
        # Used for user-defined planning algorithms. High follow mode means the robotic arm
        # doesn't optimize the path and executes the issued waypoints directly.
        # Low follow mode means the robotic arm performs interpolation on the issued path points
        # to ensure smooth execution.
        ret = self.arm.rm_movej_canfd(joints, follow)  # Uncertain if high or low follow mode
        return ret
    
    def get_joints_angles_and_pose(self):
        res = self.arm.rm_get_current_arm_state()
        joints = [res[1]['joint'][i] * M_PI / 180.0 for i in range(6)]
        pose = [res[1]['pose'][i] for i in range(6)]
        return np.array(joints), np.array(pose)
    
    def get_gripper_state(self):
        res = self.arm.rm_get_gripper_state()[1]
        gripper_pos = res['actpos'] / 1000.0

        if gripper_pos < 0.0:
            gripper_pos = 0.0
        elif gripper_pos > 1.0:
            gripper_pos = 1.0

        if res['mode'] == 4:
            gripper_speed = -GRIPPER_SPEED
        elif res['mode'] == 5:
            gripper_speed = GRIPPER_SPEED
        else:
            gripper_speed = 0.0

        return gripper_pos, gripper_speed

    def setPose_P(self, pose, speed=20, curvatureR=0, block=1):  # speed: 0-100%
        # Use joint space; can reach any reachable position. Requires planner, so startup is very brief.
        ret = self.arm.rm_movej_p(pose, speed, r=curvatureR, block=block)
        return ret

    def setPose(self, pose, speed=10, curvatureR=0, block=1):  # speed: 0-100%
        # Use Cartesian space; moves along straight line, so sometimes cannot reach certain positions.
        ret = self.arm.rm_movel(pose, speed, r=curvatureR, block=block)
        return ret

    def setPose_CANFD(self, pose, follow):  # speed: 0-100%     Uncertain if 6D pose is supported
        tag = self.arm.rm_movep_canfd(pose, follow)
        return tag

    def setJoint_CANFD(self, joint, follow):
        tag = self.arm.rm_movej_canfd(joint, follow, expand=0)
        return tag

    def setGripperPosition(self, position, block=1):
        self.arm.rm_set_gripper_position(position, block)  # position: 0-1000

import ctypes
import logging

from xrobotoolkit_teleop.hardware.interface.rm_robot_interface import *

logging.basicConfig(level=logging.INFO)
logger_ = logging.getLogger(__name__)

Joints = ctypes.c_float * 6


class my_Arm():
    def __init__(self, wifi=False):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

        if not wifi:
            self.handle = self.arm.rm_create_robot_arm("192.168.1.18", 8080)
        else:
            self.handle = self.arm.rm_create_robot_arm("192.168.33.80", 8080)

        self.joints = []
        # self.pose = Pose()

    def algo_set_joint_min_limit(self, joint_limit = [-178.0, -123.0, -178.0, -178.0, -178.0, -360.0]):
        self.arm.rm_algo_set_joint_min_limit(joint_limit = joint_limit)

    def __del__(self):
        self.arm.rm_delete_robot_arm()
        logger_.info(f'Close socket connection with robotic arm')

    def getState(self):
        res = self.arm.rm_get_current_arm_state()
        # self.joints = [_ for _ in res[1]]
        joints = [round(res[1]['joint'], 3) for i in range(6)]
        pose = [round(res[1]['pose'], 5) for i in range(6)]
        # print((joints, pose))
        return joints, pose

    def get_joints_angles_and_pose(self):
        res = self.arm.rm_get_current_arm_state()
        joints = [res[1]['joint'][i] * M_PI / 180.0 for i in range(6)]
        pose = [res[1]['pose'][i] for i in range(6)]
        return joints, pose

    def get_joint_degree(self):
        return self.arm.rm_get_joint_degree()[1]
    
    def get_gripper_state(self):
        return self.arm.rm_get_gripper_state()

    def setJoints(self, joints, v = 30, **kwargs):
        self.arm.rm_movej(joints, v = v, **kwargs)

    def setJoints_CANFD(self, joints, follow):
        ret = self.arm.rm_movej_canfd(joints, follow)  
        return ret

    def forward2Pose(self, joints):  # Unsure if 6 joints work - they do
        pose = self.arm.rm_algo_forward_kinematics(joints)
        return pose

    def inverseJoints(self, pose, block=1):  # Unsure if 6 joints work - they do
        self.getState()
        ret, joints = self.arm.rm_algo_inverse_kinematics(self.joints, pose, block)
        if ret == 0:
            return joints  # [round(joints[i],1) for i in range(6)]
        else:
            return None

    def setPose_P(self, pose, speed=20, curvatureR=0, block=1):  # speed: 0~100%
        # Use joint space, it can go where it can go. And it needs planner, so it costs a very short time to start move
        ret = self.arm.rm_movej_p(pose, speed, r=curvatureR, block=block)
        return ret

    def setPose(self, pose, speed=10, curvatureR=0, block=1):  # speed: 0~100%
        # Use Cartesian space, it goes along straight line, so sometimes it can't go someplace it can go
        ret = self.arm.rm_movel(pose, speed, r=curvatureR, block=block)
        return ret

    def setPose_CANFD(self, pose, follow):  # speed: 0~100%     Unsure if 6 poses work
        tag = self.arm.rm_movep_canfd(pose, follow)
        return tag

    def setJoint_CANFD(self, joint, follow):
        tag = self.arm.rm_movej_canfd(joint, follow, expand=0)
        return tag

    def setGripperPosition(self, position, block=1):
        self.arm.rm_set_gripper_position(position, block)  # position: 0-1000

    def gripperRelease(self, speed = 500, block = 1):
        self.arm.rm_set_gripper_release(speed, block)  # Release to max-limit

    def Get_Gripper_State(self):
        return self.arm.rm_get_gripper_state()



if __name__ == "__main__":
    arm = my_Arm(wifi=False)
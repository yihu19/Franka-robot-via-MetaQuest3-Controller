import os

import tyro

from xrobotoolkit_teleop.simulation.placo_teleop_controller import (
    PlacoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main(
    robot_urdf_path: str = os.path.join(ASSET_PATH, "realman/rml_63b/RML-63-B-gripper.urdf"),
    scale_factor: float = 1.5,
):
    """
    Main function to run the RML-63-B teleoperation with Placo.
    """
    config = {
        "right_arm": {
            "link_name": "Link_6",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",
            "gripper_config": {
              "type": "parallel",
              "gripper_trigger": "right_trigger",
              "joint_names": ["4C2_Joint1",],
              "open_pos": [0,],
              "close_pos": [0.82,],
            },
        },
    }

    # Create and initialize the teleoperation controller
    controller = PlacoTeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
    )

    # main_joints = [
    #     joint for joint in controller.placo_robot.joint_names()
    #     if joint not in ["4C2_Joint3", "4C2_Joint4", "4C2_Joint5", "4C2_Joint2", "4C2_Joint6"]
    # ]

    # additional constraints hardcoded here for now
    joints_task = controller.solver.add_joints_task()

    # for joint in controller.placo_robot.joint_names():
    #     print(f"{joint}: {controller.placo_robot.get_joint_limits(joint)}")

    joints_task.set_joints({joint: 0.2 for joint in controller.placo_robot.joint_names()})
    joints_task.configure("joints_regularization", "soft", 5e-4)

    if "joint2" in controller.placo_robot.joint_names():
        controller.placo_robot.set_joint_limits("joint2", -0.5, 0.1)  # to avoid excessive tilting of torso
        controller.solver.enable_velocity_limits(True)

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)

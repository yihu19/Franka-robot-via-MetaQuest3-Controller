import threading
import time

import tyro
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.realman_teleop_controller import ArmRealManAbsController
from xrobotoolkit_teleop.hardware.realman_increamental_controller import ArmRealManIncController

plot_controller_data=False

def main(
    reset: bool = False,
    visualize_placo: bool = False,
):
    """
    Run head and dual arm teleoperation control.

    Args:
        reset: Run the reset procedure for the dual arm controller.
        visualize_placo: Visualize Placo in the arm controller.
    """
    xr_client = XrClient()
    # head_controller = DynamixelHeadController(xr_client)
    arm_controller = ArmRealManAbsController(xr_client, visualize_placo=visualize_placo, plot_controller_data=plot_controller_data)

    if reset:
        print("Reset flag detected. Running arm controller reset procedure...")
        try:
            arm_controller.reset()
            print("Arm controller reset procedure completed.")
        except Exception as e:
            print(f"Error during arm_controller.reset(): {e}")
    else:
        print("No reset flag detected. Proceeding with normal operation.")
        arm_controller.calc_target_joint_position()

        stop_signal = threading.Event()
        # head_thread = threading.Thread(
        #     target=head_controller.run_thread,
        #     args=(stop_signal,),
        # )
        # left_arm_thread = threading.Thread(
        #     target=arm_controller.run_left_controller_thread,
        #     args=(stop_signal,),
        # )
        right_arm_thread = threading.Thread(
            target=arm_controller.run_right_controller_thread,
            args=(stop_signal,),
        )
        ik_thread = threading.Thread(
            target=arm_controller.run_ik_thread,
            args=(stop_signal,),
        )

        # Start the threads
        # head_thread.start()
        # left_arm_thread.start()
        right_arm_thread.start()
        ik_thread.start()

        if plot_controller_data:
            plot_thread = threading.Thread(
                target=arm_controller.run_plot_controller_pose_thread,
                args=(stop_signal,),
            )
            plot_thread.start()

        while not stop_signal.is_set():
            try:
                if plot_controller_data:
                    arm_controller.update_plots()  
                time.sleep(0.01)
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Exiting...")
                stop_signal.set()  # Trigger the stop signal for all threads

        if plot_controller_data:
            plot_thread.join()
        # head_thread.join()
        # left_arm_thread.join()
        right_arm_thread.join()
        ik_thread.join()

    # head_controller.close()
    arm_controller.close()
    print("All controllers have been stopped and disconnected.")

def main_inc(
    reset: bool = True,
    visualize_placo: bool = False,
):
    
    xr_client = XrClient()
    # head_controller = DynamixelHeadController(xr_client)
    arm_controller = ArmRealManIncController(xr_client, visualize_placo=visualize_placo)

    if reset:
        print("Reset flag detected. Running arm controller reset procedure...")
        try:
            arm_controller.reset()
            print("Arm controller reset procedure completed.")
        except Exception as e:
            print(f"Error during arm_controller.reset(): {e}")
    else:
        print("No reset flag detected. Proceeding with normal operation.")
        arm_controller.calc_target_joint_position()

    try:
        stop_signal = threading.Event()

        arm_control_thread = threading.Thread(
            target=arm_controller.run_ik_and_control_thread,
            args=(stop_signal,),
        )

        gripper_control_thread = threading.Thread(
            target=arm_controller.run_gripper_control_thread,
            args=(stop_signal,),
        )

        collect_state_thread = threading.Thread(
            target=arm_controller.run_collect_state_thread,
            args=(stop_signal,),
        )

        episode_control_thread = threading.Thread(
            target=arm_controller.run_episode_control_thread,
            args=(stop_signal,),
        )

        arm_control_thread.start()
        gripper_control_thread.start()
        collect_state_thread.start()
        episode_control_thread.start()

        while not stop_signal.is_set():
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Stopping control loop.")
                stop_signal.set()
        
        episode_control_thread.join()
        collect_state_thread.join()
        gripper_control_thread.join()
        arm_control_thread.join()

    except Exception as e:
        print(f"Exception during control loop: {e}")

    arm_controller.close()
    print("All controllers have been stopped and disconnected.")

if __name__ == "__main__":
    # tyro.cli(main)
    tyro.cli(main_inc)

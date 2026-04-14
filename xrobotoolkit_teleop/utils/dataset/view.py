import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True)

class RobotArmController:
    def __init__(self, xml_path):
        """Initialize the robotic arm controller"""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get joint and motor information
        self.n_joints = 6
        self.joint_names = [f'joint_{i+1}' for i in range(self.n_joints)]
        self.motor_names = [f'motor_{i+1}' for i in range(self.n_joints)]
        
        # Get joint and motor IDs
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        self.motor_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                         for name in self.motor_names]
        
        self.gripper_motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'fingers_actuator')
        self.gripper_close_position = 255  # Gripper close position
        self.gripper_open_position = 0      # Gripper open position
        
        # Data recording
        self.time_history = []
        self.position_history = []
        self.target_history = []
        
        self.viewer = None
        
    def set_target_position(self, target_pos, close_gripper=False):
        """Set target joint position"""
        self.target_positions = np.array(target_pos)
        self.data.ctrl[self.motor_ids] = self.target_positions
        target_degrees = np.degrees(self.target_positions)
        print(f"Set target: [{', '.join([f'{x:.1f}' for x in target_degrees])}]°")

        if close_gripper:
            self.data.ctrl[self.gripper_motor_id] = self.gripper_close_position  # Close gripper
        else:
            self.data.ctrl[self.gripper_motor_id] = self.gripper_open_position  # Open gripper

    def get_joint_positions(self):
        """Get current joint positions"""
        return self.data.qpos[self.joint_ids].copy()
    
    def step_simulation(self):
        """Execute one step of simulation"""
        mujoco.mj_step(self.model, self.data)
        
        # Record data
        self.time_history.append(self.data.time)
        self.position_history.append(self.get_joint_positions())
        self.target_history.append(self.target_positions.copy())
    
    def run_with_viewer(self, target_positions=None):
        """Run simulation and display visualization"""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        if target_positions:
            target_index = 0
            current_target_set = False
            
            while self.viewer.is_running():
                # If there are more target positions and the current target is not set
                if target_index < len(target_positions) and not current_target_set:
                    print(f"\n=== Moving to Position {target_index + 1} ===")
                    self.set_target_position(target_positions[target_index])
                    current_target_set = True
                
                self.step_simulation()
                self.viewer.sync()
                time.sleep(0.001)  # Control frame rate
                
                # Check if the target position is reached (optional: add logic to automatically switch to the next target)
                if current_target_set and target_index < len(target_positions):
                    final_pos = self.get_joint_positions()
                    error = np.abs(np.array(target_positions[target_index]) - final_pos)
                    max_error = np.max(error)
                    
                    # If the error is below the threshold, switch to the next target (optional)
                    if max_error < np.radians(1.0):  # 1 degree threshold
                        print(f"Reached target {target_index + 1}, Max error: {np.degrees(max_error):.1f}°")
                        target_index += 1
                        current_target_set = False
        else:
            # Only display the current state
            while self.viewer.is_running():
                self.step_simulation()
                self.viewer.sync()
                time.sleep(0.001)
        
        self.viewer.close()
        
    def interactive_control(self):
        """Interactive control"""
        print("Interactive mode - Commands:")
        print("  q: quit")
        print("  r: reset to zero position")
        print("  s <j1> <j2> <j3> <j4> <j5> <j6>: set joint angles (degrees)")
        print("  p: print current joint positions")
        print("Close the viewer window to exit")
        
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Initialize target position
        if not hasattr(self, 'target_positions'):
            self.set_target_position([0, 0, 0, 0, 0, 0])
        
        try:
            while self.viewer.is_running():
                self.step_simulation()
                self.viewer.sync()
                
                # Simplified input handling
                try:
                    import select
                    import sys
                    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                        cmd = input().strip()
                        
                        if cmd == 'q':
                            break
                        elif cmd == 'r':
                            self.set_target_position([0, 0, 0, 0, 0, 0])
                        elif cmd.startswith('s '):
                            try:
                                angles = [np.radians(float(x)) for x in cmd.split()[1:]]
                                if len(angles) == 6:
                                    self.set_target_position(angles)
                                else:
                                    print("Need 6 angles")
                            except ValueError:
                                print("Invalid angles")
                        elif cmd == 'p':
                            pos = self.get_joint_positions()
                            pos_degrees = np.degrees(pos)
                            print(f"Current: [{', '.join([f'{x:.1f}' for x in pos_degrees])}]°")
                            
                except ImportError:
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.viewer.close()
    
    def plot_results(self):
        """Plot results"""
        if not self.time_history:
            print("No data to plot")
            return
            
        time_array = np.array(self.time_history)
        pos_array = np.array(self.position_history)
        target_array = np.array(self.target_history)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Joint Position Control')
        
        for i in range(self.n_joints):
            row, col = i // 3, i % 3
            axes[row, col].plot(time_array, np.degrees(pos_array[:, i]), label='Actual')
            axes[row, col].plot(time_array, np.degrees(target_array[:, i]), '--', label='Target')
            axes[row, col].set_title(f'Joint {i+1}')
            axes[row, col].set_ylabel('Angle (°)')
            axes[row, col].legend()
            axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.show()

# def main():
#     xml_path = "/home/lyh/arm-remote/rm_63_b_description/urdf/RML-63-B-Gripper.xml"
#     controller = RobotArmController(xml_path)
    
#     # Test preset positions
#     test_positions = [
#         [0, 0, 0, 0, 0, 0],
#         # [-0.3, 0.4, -0.1, 0.2, -0.1, 0.3],
#         # [0.5, -0.2, 0.1, -0.3, 0.2, -0.4],
#         # [-0.5, 0.3, -0.2, 0.4, -0.3, 0.5],
#         # [0.2, -0.1, 0.3, -0.4, 0.1, -0.2],
#         # [0, 0, 0, 0, 0, 0],  # Return to initial position
#     ]
#     print("Running continuous simulation. Close viewer window or press Ctrl+C to exit.")
#     controller.run_with_viewer(target_positions=test_positions)
#     controller.plot_results()

def main():
    xml_path = "/home/lyh/arm-remote/rm_63_b_description/urdf/RML-63-B-Gripper.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)
    
    while True:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

if __name__ == "__main__":
    main()
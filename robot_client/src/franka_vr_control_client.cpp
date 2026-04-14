// VR-Based Cartesian Teleoperation
// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <cmath>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <array>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>  // for memset

#include <franka/exception.h>
#include <franka/robot.h>
#include <Eigen/Dense>
#include <franka/gripper.h>

#include "examples_common.h"
#include "weighted_ik.h"
#include <ruckig/ruckig.hpp>

struct VRCommand
{
    double pos_x = 0.0, pos_y = 0.0, pos_z = 0.0;
    double quat_x = 0.0, quat_y = 0.0, quat_z = 0.0, quat_w = 1.0;
    double button_pressed = 0.0; // gripper open/close command
    double gripper_speed = 0.1;  // m/s
    double gripper_force = 20.0; // N
    double epsilon_inner = 0.005; // m
    double epsilon_outer = 0.005; // m
    bool has_valid_data = false;
};

class VRController
{
private:
    std::atomic<bool> running_{true};
    VRCommand current_vr_command_;
    std::mutex command_mutex_;

    // -----------------------------------------------------------------------
    // Networking – VR command receiver (port 8888)
    // -----------------------------------------------------------------------
    int server_socket_;
    const int PORT = 8888;

    // -----------------------------------------------------------------------
    // Networking – Robot state broadcaster (port 9091)
    // -----------------------------------------------------------------------
    int state_socket_;
    struct sockaddr_in state_addr_;
    const int STATE_PORT = 9091;
    const char* STATE_IP  = "127.0.0.1";  // publish on loopback; change to "0.0.0.0" to broadcast on all interfaces

    // VR mapping parameters
    struct VRParams
    {
        double vr_smoothing = 0.05;         // Lower = more responsive

        // Deadzones to prevent drift from small sensor noise
        double position_deadzone = 0.001;   // 1 mm
        double orientation_deadzone = 0.03; // ~1.7 degrees

        // Workspace limits to keep the robot in a safe area
        double max_position_offset = 0.75;  // 75 cm from initial position
    } params_;

    // VR Target Pose
    Eigen::Vector3d vr_target_position_;
    Eigen::Quaterniond vr_target_orientation_;

    // VR filtering state
    Eigen::Vector3d filtered_vr_position_{0, 0, 0};
    Eigen::Quaterniond filtered_vr_orientation_{1, 0, 0, 0};

    // Initial poses used as a reference frame
    Eigen::Affine3d initial_robot_pose_;
    Eigen::Vector3d initial_vr_position_{0, 0, 0};
    Eigen::Quaterniond initial_vr_orientation_{1, 0, 0, 0};
    bool vr_initialized_ = false;

    // Joint space tracking
    std::array<double, 7> current_joint_angles_;
    std::array<double, 7> neutral_joint_pose_;
    std::unique_ptr<WeightedIKSolver> ik_solver_;

    // Q7 limits
    double Q7_MIN;
    double Q7_MAX;
    bool bidexhand_;
    static constexpr double Q7_SEARCH_RANGE           = 0.5;   // ±0.5 rad search around current q7
    static constexpr double Q7_OPTIMIZATION_TOLERANCE = 1e-6;
    static constexpr int    Q7_MAX_ITERATIONS          = 100;

    // Ruckig trajectory generator for smooth joint space motion
    std::unique_ptr<ruckig::Ruckig<7>> trajectory_generator_;
    ruckig::InputParameter<7>  ruckig_input_;
    ruckig::OutputParameter<7> ruckig_output_;
    bool ruckig_initialized_ = false;

    // Gradual activation to prevent sudden movements on control start
    std::chrono::steady_clock::time_point control_start_time_;
    static constexpr double ACTIVATION_TIME_SEC = 0.5;

    // Gripper control state
    bool prev_button_pressed_ = false;
    bool gripper_is_open_     = true;  // Start assuming gripper is open

    // Actual gripper width, updated by the gripper worker thread
    std::atomic<double> gripper_width_m_{0.08};  // default open width

    // Gripper worker thread and synchronization
    std::unique_ptr<franka::Gripper> gripper_;
    std::mutex              gripper_mutex_;
    std::condition_variable gripper_cv_;
    bool gripper_requested_ = false;
    struct GripperCmd {
        bool   close         = false;
        double speed         = 0.1;
        double force         = 20.0;
        double epsilon_inner = 0.005;
        double epsilon_outer = 0.005;
    } pending_gripper_cmd_;
    std::thread         gripper_thread_;
    std::atomic<bool>   gripper_thread_running_{false};

    // Franka joint limits for responsive teleoperation
    static constexpr std::array<double, 7> MAX_JOINT_VELOCITY     = {0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0};
    static constexpr std::array<double, 7> MAX_JOINT_ACCELERATION  = {1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0};
    static constexpr std::array<double, 7> MAX_JOINT_JERK          = {3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0};
    static constexpr double CONTROL_CYCLE_TIME = 0.001;  // 1 kHz

public:
    VRController(bool bidexhand = true)
        : Q7_MIN(bidexhand ? -0.2 : -2.89),
          Q7_MAX(bidexhand ?  1.9 :  2.89),
          bidexhand_(bidexhand)
    {
        setupNetworking();
        setupStateBroadcast();
    }

    ~VRController()
    {
        running_ = false;

        // Stop gripper worker
        gripper_thread_running_ = false;
        gripper_cv_.notify_one();
        if (gripper_thread_.joinable()) {
            gripper_thread_.join();
        }

        close(server_socket_);
        close(state_socket_);
    }

    // -----------------------------------------------------------------------
    // Setup – VR command UDP receiver
    // -----------------------------------------------------------------------
    void setupNetworking()
    {
        server_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (server_socket_ < 0) {
            throw std::runtime_error("Failed to create VR command socket");
        }

        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family      = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port        = htons(PORT);

        if (bind(server_socket_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            throw std::runtime_error("Failed to bind VR command socket");
        }

        int flags = fcntl(server_socket_, F_GETFL, 0);
        fcntl(server_socket_, F_SETFL, flags | O_NONBLOCK);

        std::cout << "UDP server listening on port " << PORT
                  << " for VR pose and gripper control data" << std::endl;
    }

    // -----------------------------------------------------------------------
    // Setup – Robot state UDP broadcaster
    // -----------------------------------------------------------------------
    void setupStateBroadcast()
    {
        state_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (state_socket_ < 0) {
            throw std::runtime_error("Failed to create state broadcast socket");
        }

        memset(&state_addr_, 0, sizeof(state_addr_));
        state_addr_.sin_family = AF_INET;
        state_addr_.sin_port   = htons(STATE_PORT);
        if (inet_pton(AF_INET, STATE_IP, &state_addr_.sin_addr) <= 0) {
            throw std::runtime_error("Invalid STATE_IP address");
        }

        std::cout << "Robot state will be broadcast to "
                  << STATE_IP << ":" << STATE_PORT << std::endl;
    }

    // -----------------------------------------------------------------------
    // broadcastRobotState – called from the 1 kHz control callback (throttled)
    //
    // JSON keys match the Python FrankaRobot class in franka_robot.py:
    //   robot0_joint_pos        ← rs.q
    //   robot0_joint_vel        ← rs.dq
    //   robot0_eef_pos          ← rs.O_T_EE translation
    //   robot0_eef_quat         ← rs.O_T_EE rotation → [qx,qy,qz,qw]
    //   robot0_gripper_qpos     ← gripper_width_m_ (updated by worker thread)
    //   robot0_joint_ext_torque ← rs.tau_ext_hat_filtered
    //   robot0_force_ee         ← rs.O_F_ext_hat_K[0:3]
    //   robot0_torque_ee        ← rs.O_F_ext_hat_K[3:6]
    // -----------------------------------------------------------------------
    void broadcastRobotState(const franka::RobotState& rs)
    {
        // --- EE position from column-major 4×4 transform O_T_EE ---
        // Layout: O_T_EE[0..3] = col-0, [4..7] = col-1, [8..11] = col-2, [12..15] = col-3
        double ex = rs.O_T_EE[12];
        double ey = rs.O_T_EE[13];
        double ez = rs.O_T_EE[14];

        // --- Rotation matrix → quaternion ---
        // Column-major: R(row,col) = O_T_EE[col*4 + row]
        Eigen::Matrix3d R;
        R << rs.O_T_EE[0], rs.O_T_EE[4], rs.O_T_EE[8],
             rs.O_T_EE[1], rs.O_T_EE[5], rs.O_T_EE[9],
             rs.O_T_EE[2], rs.O_T_EE[6], rs.O_T_EE[10];
        Eigen::Quaterniond eq(R);
        eq.normalize();

        // --- Gripper width from atomic updated by worker thread ---
        double gw = gripper_width_m_.load();

        char buf[2048];
        int n = snprintf(buf, sizeof(buf),
            "{"
            "\"robot0_joint_pos\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
            "\"robot0_joint_vel\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
            "\"robot0_eef_pos\":[%.6f,%.6f,%.6f],"
            "\"robot0_eef_quat\":[%.6f,%.6f,%.6f,%.6f],"
            "\"robot0_gripper_qpos\":%.6f,"
            "\"robot0_joint_ext_torque\":[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f],"
            "\"robot0_force_ee\":[%.6f,%.6f,%.6f],"
            "\"robot0_torque_ee\":[%.6f,%.6f,%.6f]"
            "}\n",
            // joint positions
            rs.q[0], rs.q[1], rs.q[2], rs.q[3], rs.q[4], rs.q[5], rs.q[6],
            // joint velocities
            rs.dq[0], rs.dq[1], rs.dq[2], rs.dq[3], rs.dq[4], rs.dq[5], rs.dq[6],
            // EE position
            ex, ey, ez,
            // EE quaternion in scipy convention [qx, qy, qz, qw]
            eq.x(), eq.y(), eq.z(), eq.w(),
            // gripper width [m]
            gw,
            // external joint torques (tau_ext_hat_filtered)
            rs.tau_ext_hat_filtered[0], rs.tau_ext_hat_filtered[1],
            rs.tau_ext_hat_filtered[2], rs.tau_ext_hat_filtered[3],
            rs.tau_ext_hat_filtered[4], rs.tau_ext_hat_filtered[5],
            rs.tau_ext_hat_filtered[6],
            // EE external force [fx, fy, fz]
            rs.O_F_ext_hat_K[0], rs.O_F_ext_hat_K[1], rs.O_F_ext_hat_K[2],
            // EE external torque [tx, ty, tz]
            rs.O_F_ext_hat_K[3], rs.O_F_ext_hat_K[4], rs.O_F_ext_hat_K[5]
        );

        sendto(state_socket_, buf, n, 0,
               (struct sockaddr*)&state_addr_, sizeof(state_addr_));
    }

    // -----------------------------------------------------------------------
    // networkThread – receives VR pose commands at ~1 kHz (non-blocking socket)
    // -----------------------------------------------------------------------
    void networkThread()
    {
        char buffer[1024];
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        while (running_)
        {
            ssize_t bytes_received = recvfrom(server_socket_, buffer, sizeof(buffer), 0,
                                              (struct sockaddr*)&client_addr, &client_len);

            if (bytes_received > 0)
            {
                buffer[bytes_received] = '\0';

                VRCommand cmd;
                int parsed_count = sscanf(buffer,
                    "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &cmd.pos_x,        &cmd.pos_y,          &cmd.pos_z,
                    &cmd.quat_x,       &cmd.quat_y,         &cmd.quat_z,      &cmd.quat_w,
                    &cmd.button_pressed, &cmd.gripper_speed, &cmd.gripper_force,
                    &cmd.epsilon_inner, &cmd.epsilon_outer);

                if (parsed_count == 12)
                {
                    cmd.has_valid_data = true;

                    std::lock_guard<std::mutex> lock(command_mutex_);
                    current_vr_command_ = cmd;

                    if (!vr_initialized_)
                    {
                        initial_vr_position_    = Eigen::Vector3d(cmd.pos_x, cmd.pos_y, cmd.pos_z);
                        initial_vr_orientation_ = Eigen::Quaterniond(
                            cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z).normalized();

                        filtered_vr_position_    = initial_vr_position_;
                        filtered_vr_orientation_ = initial_vr_orientation_;

                        vr_initialized_ = true;
                        std::cout << "VR reference pose initialized!" << std::endl;
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    // -----------------------------------------------------------------------
    // updateVRTargets – maps filtered VR delta pose onto robot target pose
    // -----------------------------------------------------------------------
    void updateVRTargets(const VRCommand& cmd)
    {
        if (!cmd.has_valid_data || !vr_initialized_) {
            return;
        }

        Eigen::Vector3d   vr_pos(cmd.pos_x, cmd.pos_y, cmd.pos_z);
        Eigen::Quaterniond vr_quat(cmd.quat_w, cmd.quat_x, cmd.quat_y, cmd.quat_z);
        vr_quat.normalize();

        // Exponential smoothing
        double alpha = 1.0 - params_.vr_smoothing;
        filtered_vr_position_    = params_.vr_smoothing * filtered_vr_position_ + alpha * vr_pos;
        filtered_vr_orientation_ = filtered_vr_orientation_.slerp(alpha, vr_quat);

        // Delta from initial VR pose
        Eigen::Vector3d    vr_pos_delta  = filtered_vr_position_ - initial_vr_position_;
        Eigen::Quaterniond vr_quat_delta = filtered_vr_orientation_ * initial_vr_orientation_.inverse();

        // Position deadzone
        if (vr_pos_delta.norm() < params_.position_deadzone) {
            vr_pos_delta.setZero();
        }
        // Orientation deadzone
        double rotation_angle = 2.0 * acos(std::abs(vr_quat_delta.w()));
        if (rotation_angle < params_.orientation_deadzone) {
            vr_quat_delta.setIdentity();
        }

        // Workspace limit
        if (vr_pos_delta.norm() > params_.max_position_offset) {
            vr_pos_delta = vr_pos_delta.normalized() * params_.max_position_offset;
        }

        vr_target_position_    = initial_robot_pose_.translation() + vr_pos_delta;
        vr_target_orientation_ = vr_quat_delta * Eigen::Quaterniond(initial_robot_pose_.rotation());
        vr_target_orientation_.normalize();
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    double clampQ7(double q7) const {
        return std::max(Q7_MIN, std::min(Q7_MAX, q7));
    }

    std::array<double, 3> eigenToArray3(const Eigen::Vector3d& vec) const {
        return {vec.x(), vec.y(), vec.z()};
    }

    std::array<double, 9> quaternionToRotationArray(const Eigen::Quaterniond& quat) const {
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        return {rot(0,0), rot(0,1), rot(0,2),
                rot(1,0), rot(1,1), rot(1,2),
                rot(2,0), rot(2,1), rot(2,2)};
    }

public:
    // -----------------------------------------------------------------------
    // run – top-level entry point
    // -----------------------------------------------------------------------
    void run(const std::string& robot_ip)
    {
        try
        {
            franka::Robot robot(robot_ip);
            setDefaultBehavior(robot);

            // --- Gripper init ---
            try {
                gripper_ = std::make_unique<franka::Gripper>(robot_ip);
                std::cout << "Gripper initialized" << std::endl;
                gripper_->homing();
                std::cout << "Gripper homed successfully" << std::endl;
                // Read actual initial width
                franka::GripperState gs = gripper_->readOnce();
                gripper_width_m_.store(gs.width);
            } catch (const std::exception& e) {
                std::cerr << "Warning: could not initialize/home gripper: " << e.what() << std::endl;
            }

            // --- Gripper worker thread ---
            // Executes blocking grasp/move commands off the RT loop.
            // After each command it reads back the actual gripper width.
            gripper_thread_running_ = true;
            gripper_thread_ = std::thread([this]() {
                while (gripper_thread_running_) {
                    GripperCmd cmd;
                    {
                        std::unique_lock<std::mutex> lk(gripper_mutex_);
                        gripper_cv_.wait(lk, [this]() {
                            return !gripper_thread_running_ || gripper_requested_;
                        });
                        if (!gripper_thread_running_) break;
                        cmd = pending_gripper_cmd_;
                        gripper_requested_ = false;
                    }
                    try {
                        if (!gripper_) {
                            std::cerr << "Gripper not initialized, skipping command" << std::endl;
                            continue;
                        }
                        if (cmd.close) {
                            gripper_->grasp(0.02, cmd.speed, cmd.force,
                                            cmd.epsilon_inner, cmd.epsilon_outer);
                        } else {
                            gripper_->move(0.08, cmd.speed);
                        }
                        // Read back actual width and store atomically for broadcaster
                        franka::GripperState gs = gripper_->readOnce();
                        gripper_width_m_.store(gs.width);
                    } catch (const franka::Exception& e) {
                        std::cerr << "Gripper worker franka exception: " << e.what() << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Gripper worker std::exception: " << e.what() << std::endl;
                    }
                }
            });

            // --- Move to starting joint configuration ---
            std::array<double, 7> q_goal;
            if (bidexhand_) {
                q_goal = {{0.0, -0.812, -0.123, -2.0, 0.0, 2.8, 0.9}};  // BiDexHand pose
            } else {
                q_goal = {{0.0, -0.48,  0.0,   -2.0, 0.0, 1.57, -0.85}}; // Full range pose
            }
            MotionGenerator motion_generator(0.5, q_goal);
            std::cout << "WARNING: This example will move the robot! "
                      << "Please make sure to have the user stop button at hand!" << std::endl
                      << "Press Enter to continue..." << std::endl;
            std::cin.ignore();
            robot.control(motion_generator);
            std::cout << "Finished moving to initial joint configuration." << std::endl;

            // --- Collision behavior ---
            robot.setCollisionBehavior(
                {{100.0,100.0,80.0,80.0,80.0,80.0,60.0}},
                {{100.0,100.0,80.0,80.0,80.0,80.0,60.0}},
                {{100.0,100.0,80.0,80.0,80.0,80.0,60.0}},
                {{100.0,100.0,80.0,80.0,80.0,80.0,60.0}},
                {{80.0,80.0,80.0,80.0,80.0,80.0}},
                {{80.0,80.0,80.0,80.0,80.0,80.0}},
                {{80.0,80.0,80.0,80.0,80.0,80.0}},
                {{80.0,80.0,80.0,80.0,80.0,80.0}});

            // --- Joint impedance ---
            robot.setJointImpedance({{3000,3000,3000,2500,2500,2000,2000}});

            // --- Read initial robot state ---
            franka::RobotState state = robot.readOnce();
            initial_robot_pose_ = Eigen::Affine3d(Eigen::Matrix4d::Map(state.O_T_EE.data()));

            for (int i = 0; i < 7; i++) {
                current_joint_angles_[i] = state.q[i];
                neutral_joint_pose_[i]   = q_goal[i];
            }

            // --- IK solver ---
            std::array<double, 7> base_joint_weights = {{
                3.0,  // Joint 0 – base rotation:  high penalty for stability
                6.0,  // Joint 1 – base shoulder:  high penalty for stability
                1.5,  // Joint 2 – elbow
                1.5,  // Joint 3 – forearm
                1.0,  // Joint 4 – wrist
                1.0,  // Joint 5 – wrist
                1.0   // Joint 6 – hand
            }};
            ik_solver_ = std::make_unique<WeightedIKSolver>(
                neutral_joint_pose_,
                1.0,   // manipulability weight
                2.0,   // neutral distance weight
                2.0,   // current distance weight
                base_joint_weights,
                false  // verbose = false for production use
            );

            // --- Ruckig ---
            trajectory_generator_ = std::make_unique<ruckig::Ruckig<7>>();
            trajectory_generator_->delta_time = CONTROL_CYCLE_TIME;
            for (size_t i = 0; i < 7; ++i) {
                ruckig_input_.max_velocity[i]     = MAX_JOINT_VELOCITY[i];
                ruckig_input_.max_acceleration[i] = MAX_JOINT_ACCELERATION[i];
                ruckig_input_.max_jerk[i]         = MAX_JOINT_JERK[i];
                ruckig_input_.target_velocity[i]     = 0.0;
                ruckig_input_.target_acceleration[i] = 0.0;
            }
            std::cout << "Ruckig trajectory generator configured with 7 DOFs" << std::endl;

            // --- Initial VR targets = robot's starting pose ---
            vr_target_position_    = initial_robot_pose_.translation();
            vr_target_orientation_ = Eigen::Quaterniond(initial_robot_pose_.rotation());

            // --- Start network thread and wait for VR data ---
            // While waiting, broadcast robot state so the Python data-collection
            // script can receive valid state immediately, independent of whether
            // the VR headset has sent its first packet yet.
            std::thread network_thread(&VRController::networkThread, this);
            std::cout << "Waiting for VR data..." << std::endl;
            while (!vr_initialized_ && running_) {
                // Read current robot state and broadcast it at ~10 Hz
                try {
                    franka::RobotState pre_state = robot.readOnce();
                    broadcastRobotState(pre_state);
                } catch (const franka::Exception& e) {
                    std::cerr << "Warning: readOnce failed during VR wait: " << e.what() << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (vr_initialized_) {
                std::cout << "VR initialized! Starting real-time control." << std::endl;
                this->runVRControl(robot);
            }

            running_ = false;
            if (network_thread.joinable())
                network_thread.join();

            gripper_thread_running_ = false;
            gripper_cv_.notify_one();
            if (gripper_thread_.joinable())
                gripper_thread_.join();
        }
        catch (const franka::Exception& e)
        {
            std::cerr << "Franka exception: " << e.what() << std::endl;
            running_ = false;
        }
    }

private:
    // -----------------------------------------------------------------------
    // runVRControl – real-time joint velocity control loop (1 kHz)
    // -----------------------------------------------------------------------
    void runVRControl(franka::Robot& robot)
    {
        auto vr_control_callback = [this](
            const franka::RobotState& robot_state,
            franka::Duration           period) -> franka::JointVelocities
        {
            // --- Update VR targets from latest command ---
            VRCommand cmd;
            {
                std::lock_guard<std::mutex> lock(command_mutex_);
                cmd = current_vr_command_;
            }
            updateVRTargets(cmd);

            // --- Gripper toggle on button press edge ---
            bool button_pressed = cmd.button_pressed > 0.5;
            if (button_pressed && !prev_button_pressed_) {
                {
                    std::lock_guard<std::mutex> lk(gripper_mutex_);
                    pending_gripper_cmd_.close         = gripper_is_open_;
                    pending_gripper_cmd_.speed         = cmd.gripper_speed;
                    pending_gripper_cmd_.force         = cmd.gripper_force;
                    pending_gripper_cmd_.epsilon_inner = cmd.epsilon_inner;
                    pending_gripper_cmd_.epsilon_outer = cmd.epsilon_outer;
                    gripper_requested_ = true;
                }
                gripper_cv_.notify_one();
                gripper_is_open_ = !gripper_is_open_;
            }
            prev_button_pressed_ = button_pressed;

            // --- Ruckig initialization (first call only) ---
            if (!ruckig_initialized_) {
                for (int i = 0; i < 7; i++) {
                    current_joint_angles_[i]            = robot_state.q[i];
                    ruckig_input_.current_position[i]    = robot_state.q[i];
                    ruckig_input_.current_velocity[i]    = 0.0;
                    ruckig_input_.current_acceleration[i]= 0.0;
                    ruckig_input_.target_position[i]     = robot_state.q[i];
                    ruckig_input_.target_velocity[i]     = 0.0;
                }
                control_start_time_ = std::chrono::steady_clock::now();
                ruckig_initialized_ = true;
                std::cout << "Ruckig initialized – starting with zero velocity commands." << std::endl;
            } else {
                // Use Ruckig's own previous output for continuity (avoid encoder noise)
                for (int i = 0; i < 7; i++) {
                    current_joint_angles_[i]             = robot_state.q[i];
                    ruckig_input_.current_position[i]    = robot_state.q[i];
                    ruckig_input_.current_velocity[i]    = ruckig_output_.new_velocity[i];
                    ruckig_input_.current_acceleration[i]= ruckig_output_.new_acceleration[i];
                }
            }

            // --- Gradual activation factor ---
            auto   current_time   = std::chrono::steady_clock::now();
            double elapsed_sec    = std::chrono::duration<double>(current_time - control_start_time_).count();
            double activation_factor = std::min(1.0, elapsed_sec / ACTIVATION_TIME_SEC);

            // --- IK solve ---
            std::array<double, 3> target_pos = eigenToArray3(vr_target_position_);
            std::array<double, 9> target_rot = quaternionToRotationArray(vr_target_orientation_);

            double current_q7 = current_joint_angles_[6];
            double q7_start   = std::max(-2.89, current_q7 - Q7_SEARCH_RANGE);
            double q7_end     = std::min( 2.89, current_q7 + Q7_SEARCH_RANGE);

            WeightedIKResult ik_result = ik_solver_->solve_q7_optimized(
                target_pos, target_rot, current_joint_angles_,
                q7_start, q7_end, Q7_OPTIMIZATION_TOLERANCE, Q7_MAX_ITERATIONS);

            // --- Debug printout (every 100 ms) ---
            static int debug_counter = 0;
            debug_counter++;
            if (debug_counter % 100 == 0) {
                std::cout << "IK: "
                          << (ik_result.success ? "\033[32msuccess\033[0m" : "\033[31mfail\033[0m")
                          << " | Joints: ";
                for (int i = 0; i < 7; i++) {
                    std::cout << std::fixed << std::setprecision(2) << current_joint_angles_[i];
                    if (i < 6) std::cout << " ";
                }
                std::cout << std::endl;
            }

            // --- Update Ruckig targets ---
            if (ik_result.success) {
                for (int i = 0; i < 7; i++) {
                    double current_pos    = current_joint_angles_[i];
                    double ik_target_pos  = ik_result.joint_angles[i];
                    ruckig_input_.target_position[i] =
                        current_pos + activation_factor * (ik_target_pos - current_pos);
                    ruckig_input_.target_velocity[i] = 0.0;
                }
                // Enforce BiDexHand / full-range q7 limits
                ruckig_input_.target_position[6] = clampQ7(ruckig_input_.target_position[6]);
            }
            // On IK failure: keep previous targets (Ruckig will hold/decelerate)

            // --- Ruckig update ---
            ruckig::Result ruckig_result =
                trajectory_generator_->update(ruckig_input_, ruckig_output_);

            std::array<double, 7> target_joint_velocities;
            if (ruckig_result == ruckig::Result::Working ||
                ruckig_result == ruckig::Result::Finished) {
                for (int i = 0; i < 7; i++) {
                    target_joint_velocities[i] = ruckig_output_.new_velocity[i];
                }
            } else {
                // Emergency stop
                target_joint_velocities.fill(0.0);
                if (debug_counter % 100 == 0) {
                    std::cout << "Ruckig error – zero velocity for safety." << std::endl;
                }
            }

            // --- Broadcast robot state at ~100 Hz (every 10 control cycles) ---
            if (debug_counter % 10 == 0) {
                broadcastRobotState(robot_state);
            }

            if (!running_) {
                return franka::MotionFinished(
                    franka::JointVelocities({0.0,0.0,0.0,0.0,0.0,0.0,0.0}));
            }
            return franka::JointVelocities(target_joint_velocities);
        };

        try {
            robot.control(vr_control_callback);
        } catch (const franka::ControlException& e) {
            std::cerr << "VR control exception: " << e.what() << std::endl;
        }
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <robot-hostname> [bidexhand]" << std::endl;
        std::cerr << "  bidexhand: true (default) for BiDexHand limits, false for full range"
                  << std::endl;
        return -1;
    }

    bool bidexhand = false;
    if (argc == 3) {
        std::string arg = argv[2];
        bidexhand = (arg == "true" || arg == "1");
    }

    try {
        VRController controller(bidexhand);
        controller.run(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
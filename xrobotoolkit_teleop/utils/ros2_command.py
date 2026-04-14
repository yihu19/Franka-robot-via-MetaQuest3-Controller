import threading
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState as RosJointState

class Ros2CollectorClient:
    """
    Responsible for:
      - Publishing episode control commands to /episode_control
      - Periodically publishing /robot_state as needed (start_collect/stop_collect only control the timer)
    """
    def __init__(
        self, collect_topic: str = "/collect_client", 
        episode_topic: str = "/episode_control", 
        reliable: bool = True, 
        node_name: str = "teleop_collect_client", 
        mode="internal"
        ):
        if rclpy is None:
            raise RuntimeError("rclpy is not installed or unavailable, cannot use Ros2CollectorClient.")
        # collect_topic is deprecated, kept for compatibility with constructor signature
        self._topic_episode = episode_topic
        self._topic_collect = collect_topic  # New: collection control topic
        self._node_name = node_name
        self._lock = threading.Lock()
        self._node = None
        self._episode_pub = None
        self._collect_pub = None  # New
        self._mode = mode.lower().strip()

        reliability = ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
        self._qos = QoSProfile(reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=1)

        self._executor = None
        self._spin_thread = None

        self._state_provider = None  # callable -> JointState | dict | tuple | list
        self._state_topic = "/robot_state"
        self._state_rate_hz = 10.0
        self._state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._state_pub = None
        self._state_timer = None

    def _ensure_node(self):
        with self._lock:
            if self._mode == "internal" and self._node is not None and self._episode_pub is not None and self._executor is not None:
                return
            
            if self._mode == "external" and self._node is not None and self._collect_pub is not None and self._executor is not None:
                return
            
            if not rclpy.ok():
                try:
                    rclpy.init(args=None)
                except Exception:
                    pass

            try:
                if self._node is None:
                    self._node = rclpy.create_node(self._node_name)
                if self._mode == "internal" and self._episode_pub is None:
                    self._episode_pub = self._node.create_publisher(String, self._topic_episode, self._qos)
                    print(f"[ROS2] Node {self._node_name} publishing to: {self._topic_episode}")

                if self._mode == "external" and self._collect_pub is None:
                    self._collect_pub = self._node.create_publisher(String, self._topic_collect, self._qos)
                    print(f"[ROS2] Node {self._node_name} publishing to: {self._topic_collect}")

            except Exception:
                # Retry once
                self._node = rclpy.create_node(self._node_name)
                if self._mode == "internal":
                    self._episode_pub = self._node.create_publisher(String, self._topic_episode, self._qos)
                if self._mode == "external":
                    self._collect_pub = self._node.create_publisher(String, self._topic_collect, self._qos)
                print(f"[Recovered] publishers ready -> {self._topic_episode}, {self._topic_collect}")

            if self._executor is None:
                self._executor = SingleThreadedExecutor()
                self._executor.add_node(self._node)
                def _spin():
                    try:
                        self._executor.spin()
                    except Exception:
                        pass
                self._spin_thread = threading.Thread(target=_spin, daemon=True)
                self._spin_thread.start()

    def _ensure_state_pub(self):
        if self._state_provider is None:
            return
        self._ensure_node()
        with self._lock:
            if self._state_pub is None:
                self._state_pub = self._node.create_publisher(RosJointState, self._state_topic, self._state_qos)
                self._node.get_logger().info(f"[Collector] state publisher ready -> {self._state_topic} @ {self._state_rate_hz}Hz")

    def set_state_provider(self, provider, topic: str = "/robot_state", rate_hz: float = 20.0, reliable: bool = True):
        self._state_provider = provider
        self._state_topic = topic
        self._state_rate_hz = max(0.1, float(rate_hz))
        reliability = ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
        self._state_qos = QoSProfile(reliability=reliability, history=HistoryPolicy.KEEP_LAST, depth=10)

    def _start_state_timer(self):
        if self._state_provider is None:
            return
        self._ensure_state_pub()
        with self._lock:
            if self._state_timer is None:
                period = 1.0 / self._state_rate_hz
                self._state_timer = self._node.create_timer(period, self._on_state_timer)
                self._node.get_logger().info(f"[Collector] state timer started ({self._state_rate_hz} Hz)")

    def _stop_state_timer(self):
        with self._lock:
            try:
                if self._state_timer is not None and self._node is not None:
                    self._node.destroy_timer(self._state_timer)
                    self._state_timer = None
                    self._node.get_logger().info("[Collector] state timer stopped")
            except Exception as e:
                print(f"[Collector][ROS2] Error stopping state timer: {e}")

    def _on_state_timer(self):
        try:
            data = self._state_provider() if self._state_provider else None
            if data is None or self._state_pub is None:
                return
            msg = self._to_joint_state(data)
            if msg is None:
                return
            msg.header.stamp = self._node.get_clock().now().to_msg()
            if not msg.header.frame_id:
                msg.header.frame_id = "base_link"
            self._state_pub.publish(msg)
        except Exception as e:
            try:
                self._node.get_logger().warn(f"[Collector] state provider error: {e}")
            except Exception:
                print(f"[Collector] state provider error: {e}")

    def _to_joint_state(self, data):
        if isinstance(data, RosJointState):
            return data
        msg = RosJointState()
        if isinstance(data, dict):
            msg.name = list(data.get("name", []))
            msg.position = list(data.get("position", []))
            if "velocity" in data: msg.velocity = list(data["velocity"])
            if "effort" in data: msg.effort = list(data["effort"])
            msg.header.frame_id = data.get("frame_id", "")
            return msg
        if isinstance(data, tuple) and len(data) == 2:
            names, positions = data
            msg.name = list(names)
            msg.position = list(positions)
            return msg
        if isinstance(data, (list, tuple)):
            positions = list(data)
            msg.name = [f"joint{i+1}" for i in range(len(positions))]
            msg.position = positions
            return msg
        return None

    # No longer publishing collect_cmd, only controlling the state publishing timer
    def start_collect(self):
        try:
            self._start_state_timer()
        except Exception as e:
            print(f"[Collector][ROS2] start_collect failed: {e}")

    def stop_collect(self):
        try:
            self._stop_state_timer()
        except Exception as e:
            print(f"[Collector][ROS2] stop_collect failed: {e}")

    # external mode: only send collection control commands to robot_pub (distinguished from episode_control)
    def send_collect_cmd(self, cmd: str):
        self._ensure_node()
        if self._collect_pub is None:
            return
        msg = String()
        msg.data = (cmd or "").strip().upper()
        self._collect_pub.publish(msg)

    def start_collect_cmd(self):
        self.send_collect_cmd("START")

    def stop_collect_cmd(self):
        self.send_collect_cmd("STOP")

    # Send episode control commands
    def send_episode_control(self, cmd: str):
        self._ensure_node()
        msg = String()
        msg.data = (cmd or "").strip().upper()
        self._episode_pub.publish(msg)

    def start_episode(self):
        self.send_episode_control("START")

    def stop_episode(self):
        self.send_episode_control("STOP")

    def close(self):
        with self._lock:
            try:
                if self._state_timer is not None and self._node is not None:
                    self._node.destroy_timer(self._state_timer); self._state_timer = None
                if self._state_pub is not None and self._node is not None:
                    self._node.destroy_publisher(self._state_pub); self._state_pub = None
                if self._episode_pub is not None and self._node is not None:
                    self._node.destroy_publisher(self._episode_pub); self._episode_pub = None
                if self._collect_pub is not None and self._node is not None:
                    self._node.destroy_publisher(self._collect_pub); self._collect_pub = None

                if self._executor is not None:
                    try: self._executor.shutdown(timeout_sec=1.0)
                    except Exception: pass
                    self._executor = None

                if self._spin_thread is not None:
                    try: self._spin_thread.join(timeout=1.0)
                    except Exception: pass
                    self._spin_thread = None

                if self._node is not None:
                    self._node.destroy_node(); self._node = None
            except Exception as e:
                print(f"[Collector][ROS2] Error closing resources: {e}")
        try:
            if hasattr(self, "_ep_thread") and self._ep_thread is not None:
                self._ep_thread.join(timeout=0.5)
        except Exception:
            pass
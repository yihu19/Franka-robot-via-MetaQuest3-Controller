"""
game_controller.py – Meta Quest 3 button-based recording control.

Listens on a local UDP socket for "START" / "STOP" signals broadcast by
franka_teleop_controller.py when the right-controller B / A buttons are pressed.
This avoids initialising the XRoboToolkit SDK a second time (which would steal
the connection from the teleoperation process).

    Right B → start recording   (teleop process detects press → sends "START" UDP)
    Right A → save & stop recording  (teleop process → sends "STOP" UDP)
    Ctrl-C  → quit
"""

import socket
import threading

COLLECTION_CTRL_PORT = 8765   # must match ArmFrankaIncController.COLLECTION_CTRL_PORT


class QuestController:
    """
    Receives START/STOP commands from the teleoperation process via UDP.

    get_recording_controls() returns (start, stop, quit=False).
    Each flag is True for exactly one call per received message.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_pending = False
        self._stop_pending = False

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", COLLECTION_CTRL_PORT))
        self._sock.settimeout(0.05)

        self._thread = threading.Thread(target=self._recv_loop, daemon=True, name="quest-ctrl-udp")
        self._thread.start()

        print("[QuestController] Listening for button signals from teleoperation process.")
        print(f"  UDP port {COLLECTION_CTRL_PORT}: 'START' → begin recording, 'STOP' → save & stop")
        print("  Press Right B on Quest 3 to start, Right A to stop.")
        print("  Ctrl-C to quit.")

    def _recv_loop(self):
        while True:
            try:
                data, _ = self._sock.recvfrom(64)
                cmd = data.decode("utf-8", errors="ignore").strip().upper()
                with self._lock:
                    if cmd == "START":
                        self._start_pending = True
                    elif cmd == "STOP":
                        self._stop_pending = True
            except socket.timeout:
                continue
            except Exception:
                break

    def get_recording_controls(self):
        """
        Returns (start_recording, stop_recording, quit=False).
        Flags are cleared after reading (fire-once semantics).
        """
        with self._lock:
            start = self._start_pending
            stop = self._stop_pending
            self._start_pending = False
            self._stop_pending = False
        return start, stop, False

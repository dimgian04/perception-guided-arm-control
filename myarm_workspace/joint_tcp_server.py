"""Standalone joint-target TCP server for the myArm Raspberry Pi.

De-ROS-ified version of the old ``joint_target_tcp_server`` node: it speaks the
exact same JSON-line TCP protocol, but talks to the arm directly over serial via
pymycobot with no rclpy/ROS dependency.

Protocol (one JSON object per line in, one JSON object per line out):
    {"type": "get_angles"}                     -> current joint angles (deg/rad)
    {"angles_deg": [...7...], "speed": 20}      -> move to those joint angles
    {"angles_rad": [...7...]}                   -> move (radians accepted too)

Run on the Pi:
    python3 joint_tcp_server.py
    python3 joint_tcp_server.py --port-serial /dev/ttyAMA0 --bind-port 5017
"""

import argparse
import json
import math
import socket
import time

try:
    from pymycobot.myarm import MyArm
except ImportError:
    MyArm = None


JOINT_COUNT = 7


def log(message):
    print(f"[joint-tcp] {time.strftime('%H:%M:%S')} {message}", flush=True)


class JointTargetTcpServer:
    def __init__(self, args):
        self.bind_host = args.bind_host
        self.bind_port = int(args.bind_port)
        self.speed = int(args.speed)
        self.duplicate_tolerance_deg = float(args.duplicate_tolerance_deg)
        self.dry_run = bool(args.dry_run) or MyArm is None

        self.last_target_deg = None
        self.server_socket = None
        self.fake_angles = [0.0] * JOINT_COUNT  # used only in dry-run

        if self.dry_run:
            log("DRY-RUN: not connecting to myArm serial port.")
            self.myarm = None
        else:
            log(f"Connecting to myArm on {args.port_serial} at {args.baudrate} baud")
            self.myarm = MyArm(port=args.port_serial, baudrate=int(args.baudrate), timeout=1.0)
            time.sleep(0.2)
            self.myarm.set_fresh_mode(mode=1)
            time.sleep(0.2)

        if bool(args.home_on_start):
            log("Moving myArm to home joint angles [0, 0, 0, 0, 0, 0, 0]")
            self.send_angles([0.0] * JOINT_COUNT, self.speed)
            time.sleep(1.0)

    # ------------------------------------------------------------
    # Arm I/O (real or dry-run stub).
    # ------------------------------------------------------------

    def send_angles(self, angles_deg, speed):
        if self.dry_run:
            self.fake_angles = [float(a) for a in angles_deg]
            return
        self.myarm.send_angles(angles_deg, speed)

    def read_current_angles(self):
        if self.dry_run:
            return [float(a) for a in self.fake_angles]

        current_angles = None
        last_error = None
        for method_name in ("get_angles", "get_joints_angle"):
            method = getattr(self.myarm, method_name, None)
            if method is None:
                continue
            try:
                current_angles = method()
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if current_angles is None and last_error is not None:
            log(f"Could not read myArm joint angles: {last_error}")
            return None
        if current_angles is None or len(current_angles) != JOINT_COUNT:
            log(f"Invalid current joint angles from myArm: {current_angles}")
            return None
        return [float(angle) for angle in current_angles]

    # ------------------------------------------------------------
    # TCP server.
    # ------------------------------------------------------------

    def serve_forever(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self.server_socket = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.bind_host, self.bind_port))
            server.listen(5)
            server.settimeout(0.5)
            log(f"myArm joint TCP server listening on {self.bind_host}:{self.bind_port}")

            while True:
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with client:
                    client.settimeout(2.0)
                    self.handle_client(client, address)

    def handle_client(self, client, address):
        try:
            line = self.read_line(client)
            command = json.loads(line)
            result = self.handle_command(command, address)
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": str(exc)}
        try:
            client.sendall((json.dumps(result) + "\n").encode("utf-8"))
        except OSError:
            pass

    def handle_command(self, command, address):
        command_type = str(command.get("type", "")).lower()
        if command_type in ("get_angles", "read_angles", "current_angles") or command.get("get_angles", False):
            return self.handle_get_angles(command, address)

        angles_deg = self.extract_angles_deg(command)
        speed = int(command.get("speed", self.speed))
        speed = max(1, min(speed, 100))

        if self.is_duplicate_target(angles_deg):
            return {"ok": True, "duplicate": True, "seq": command.get("seq"), "angles_deg": angles_deg}

        self.last_target_deg = angles_deg
        log(f"TCP target from {address[0]}:{address[1]} seq={command.get('seq')} "
            f"deg={[round(angle, 3) for angle in angles_deg]} speed={speed}")
        self.send_angles(angles_deg, speed)
        return {
            "ok": True,
            "duplicate": False,
            "seq": command.get("seq"),
            "angles_deg": angles_deg,
            "speed": speed,
        }

    def handle_get_angles(self, command, address):
        angles_deg = self.read_current_angles()
        if angles_deg is None:
            return {"ok": False, "seq": command.get("seq"),
                    "error": "Could not read current myArm joint angles"}
        log(f"TCP current-angle request from {address[0]}:{address[1]} "
            f"seq={command.get('seq')} deg={[round(angle, 3) for angle in angles_deg]}")
        return {
            "ok": True,
            "seq": command.get("seq"),
            "angles_deg": angles_deg,
            "angles_rad": [math.radians(angle) for angle in angles_deg],
        }

    @staticmethod
    def extract_angles_deg(command):
        if "angles_deg" in command:
            angles_deg = [float(angle) for angle in command["angles_deg"]]
        elif "angles_rad" in command:
            angles_deg = [math.degrees(float(angle)) for angle in command["angles_rad"]]
        else:
            raise ValueError("Command must contain angles_deg or angles_rad")
        if len(angles_deg) != JOINT_COUNT:
            raise ValueError(f"Expected {JOINT_COUNT} joint angles, got {len(angles_deg)}")
        return angles_deg

    def is_duplicate_target(self, angles_deg):
        if self.last_target_deg is None:
            return False
        diffs = [abs(float(c) - float(t)) for c, t in zip(angles_deg, self.last_target_deg)]
        return max(diffs) <= self.duplicate_tolerance_deg

    @staticmethod
    def read_line(sock):
        chunks = []
        while True:
            chunk = sock.recv(1)
            if not chunk:
                break
            if chunk == b"\n":
                break
            chunks.append(chunk)
        if not chunks:
            raise ValueError("Empty TCP command")
        return b"".join(chunks).decode("utf-8")

    def shutdown(self):
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone myArm joint-target TCP server.")
    parser.add_argument("--bind-host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--bind-port", type=int, default=5017, help="Bind port")
    parser.add_argument("--port-serial", default="/dev/ttyAMA0", help="myArm serial port")
    parser.add_argument("--baudrate", type=int, default=115200, help="myArm serial baudrate")
    parser.add_argument("--speed", type=int, default=20, help="Default send_angles speed (1..100)")
    parser.add_argument("--home-on-start", action="store_true", default=True,
                        help="Move to all-zeros home on startup")
    parser.add_argument("--no-home-on-start", dest="home_on_start", action="store_false",
                        help="Do not move home on startup")
    parser.add_argument("--duplicate-tolerance-deg", type=float, default=0.5,
                        help="Ignore targets within this angle of the last one")
    parser.add_argument("--dry-run", action="store_true", help="Do not open the serial port (for testing off-Pi)")
    return parser.parse_args()


def main():
    args = parse_args()
    server = JointTargetTcpServer(args)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()

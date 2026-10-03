import json
import socket
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointTargetTcpBridgeNode(Node):
    def __init__(self):
        super().__init__("joint_target_tcp_bridge_node")

        self.declare_parameter("topic_name", "/myarm/joint_targets")
        self.declare_parameter("robot_host", "192.168.0.103")
        self.declare_parameter("robot_port", 5017)
        self.declare_parameter("speed", 60)
        self.declare_parameter("connect_timeout_s", 2.0)
        self.declare_parameter("ack_timeout_s", 1.0)
        self.declare_parameter("send_repeats", 3)
        self.declare_parameter("repeat_delay_s", 0.05)

        self.sequence = 0
        self.subscription = self.create_subscription(
            JointState,
            self.get_parameter("topic_name").value,
            self.joint_target_callback,
            10,
        )

        self.get_logger().info(
            "TCP bridge ready: "
            f"{self.get_parameter('topic_name').value} -> "
            f"{self.get_parameter('robot_host').value}:"
            f"{self.get_parameter('robot_port').value}"
        )

    def joint_target_callback(self, msg):
        if len(msg.position) != 7:
            self.get_logger().warn(f"Expected 7 joint angles, got {len(msg.position)}")
            return

        self.sequence += 1
        command = {
            "type": "joint_target",
            "seq": self.sequence,
            "angles_rad": [float(angle) for angle in msg.position],
            "speed": int(self.get_parameter("speed").value),
        }

        repeats = max(1, int(self.get_parameter("send_repeats").value))
        for attempt in range(repeats):
            ok = self.send_command(command)
            if ok:
                if attempt > 0:
                    self.get_logger().info(
                        f"Joint target seq={self.sequence} delivered on retry {attempt + 1}"
                    )
                return
            time.sleep(float(self.get_parameter("repeat_delay_s").value))

        self.get_logger().error(f"Failed to deliver joint target seq={self.sequence}")


    def send_command(self, command):
        host = str(self.get_parameter("robot_host").value)
        port = int(self.get_parameter("robot_port").value)
        connect_timeout = float(self.get_parameter("connect_timeout_s").value)
        ack_timeout = float(self.get_parameter("ack_timeout_s").value)

        try:
            with socket.create_connection((host, port), timeout=connect_timeout) as sock:
                sock.settimeout(ack_timeout)
                payload = (json.dumps(command) + "\n").encode("utf-8")
                sock.sendall(payload)
                ack = self.read_line(sock)
        except OSError as exc:
            self.get_logger().warn(f"TCP send failed to {host}:{port}: {exc}")
            return False

        try:
            ack_data = json.loads(ack)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Bad TCP ack: {ack!r}")
            return False

        if not ack_data.get("ok", False):
            self.get_logger().warn(f"Robot rejected command: {ack_data}")
            return False

        self.get_logger().info(
            f"Sent joint target seq={command['seq']} to myArm TCP server"
        )
        return True

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
        return b"".join(chunks).decode("utf-8")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = JointTargetTcpBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

import math
from typing import Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class MazeStateMachineNode(Node):
    GO_FORWARD = "GO_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    RECOVERY = "RECOVERY"
    STOP_BEFORE_TURN = "STOP_BEFORE_TURN"

    VALID_COLORS = {"red", "green"}

    def __init__(self) -> None:
        super().__init__("maze_state_machine")

        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            True
        )])

        self.forward_speed = float(self.declare_parameter("forward_speed", 0.18).value)
        self.turn_speed = float(self.declare_parameter("turn_speed", 0.4).value)
        self.front_obstacle_threshold = float(
            self.declare_parameter("front_obstacle_threshold", 0.45).value
        )
        self.approach_slowdown_distance = float(
            self.declare_parameter("approach_slowdown_distance", 0.8).value
        )
        self.min_approach_speed = float(
            self.declare_parameter("min_approach_speed", 0.05).value
        )
        self.lateral_reduce_distance = float(
            self.declare_parameter("lateral_reduce_distance", 1.0).value
        )
        self.lateral_reduce_factor = float(
            self.declare_parameter("lateral_reduce_factor", 0.35).value
        )
        self.marker_distance_threshold = float(
            self.declare_parameter("marker_distance_threshold", 0.5).value
        )
        self.control_rate_hz = float(self.declare_parameter("control_rate_hz", 10.0).value)

        self.target_yaw: Optional[float] = None
        self.yaw_tolerance = 0.05

        self.next_state_after_stop = ""
        self.stop_wait_duration = 0.4
        self.current_forward_speed = 0.0
        self.acceleration_step = 0.02

        self.cmd_pub = self.create_publisher(Twist, "/jetauto/cmd_vel", 10)
        self.create_subscription(LaserScan, "/jetauto/lidar/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(String, "/color_detected", self.color_callback, 10)
        self.create_subscription(Twist, "/cmd_vel_nav", self.nav_callback, 10)

        self.latest_nav_cmd = Twist()
        self.current_state = self.GO_FORWARD
        self.previous_state = ""
        self.state_end_time = 0.0

        self.latest_scan: Optional[LaserScan] = None
        self.current_position: Optional[Tuple[float, float, float]] = None
        self.latest_color: str = "none"
        self.used_markers: List[Dict[str, float]] = []
        self.last_front_distance: float = float("inf")

        self.create_timer(1.0 / self.control_rate_hz, self.control_loop)
        self.get_logger().info("Maquina de estados iniciada")

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.current_position = (x, y, yaw)

    def color_callback(self, msg: String) -> None:
        self.latest_color = msg.data.strip().lower()

    def nav_callback(self, msg: Twist) -> None:
        self.latest_nav_cmd = msg

    def log_transition(self, new_state: str, reason: str) -> None:
        if new_state != self.previous_state:
            self.get_logger().info(f"[{self.previous_state} -> {new_state}] {reason}")
            self.previous_state = new_state

    def normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def transition_to_stop_then_turn(self, next_state: str, reason: str) -> None:
        self.current_state = self.STOP_BEFORE_TURN
        self.next_state_after_stop = next_state
        self.state_end_time = self.now_seconds() + self.stop_wait_duration
        self.current_forward_speed = 0.0
        self.publish_cmd_vel(0.0, 0.0)
        self.log_transition(
            self.STOP_BEFORE_TURN,
            f"{reason} | proximo={next_state} espera={self.stop_wait_duration}s"
        )

    def control_loop(self) -> None:
        if self.latest_scan is None or self.current_position is None:
            self.publish_cmd_vel(0.0, 0.0)
            return

        x, y, current_yaw = self.current_position
        now = self.now_seconds()

        if self.current_state == self.STOP_BEFORE_TURN:
            if now >= self.state_end_time:
                self.current_state = self.next_state_after_stop
                self.log_transition(self.current_state, "parada concluida")
            else:
                self.publish_cmd_vel(0.0, 0.0)
            return

        if self.current_state in {self.TURN_LEFT, self.TURN_RIGHT}:
            if self.target_yaw is not None:
                angle_diff = abs(self.normalize_angle(self.target_yaw - current_yaw))

                if angle_diff <= self.yaw_tolerance:
                    self.target_yaw = None
                    self.current_state = self.GO_FORWARD
                    self.current_forward_speed = 0.0
                    self.latest_color = "none"
                    self.log_transition(
                        self.GO_FORWARD,
                        f"curva concluida erro_yaw={angle_diff:.3f}rad"
                    )
                else:
                    self.execute_current_state()
                    return

        if self.is_new_color_event():
            self.record_current_marker(self.latest_color)

            if self.latest_color == "red":
                self.target_yaw = self.normalize_angle(current_yaw + (math.pi / 2.0))
                self.transition_to_stop_then_turn(
                    self.TURN_LEFT,
                    f"cor=vermelho yaw_alvo={math.degrees(self.target_yaw):.1f}°"
                )

            elif self.latest_color == "green":
                self.target_yaw = self.normalize_angle(current_yaw - (math.pi / 2.0))
                self.transition_to_stop_then_turn(
                    self.TURN_RIGHT,
                    f"cor=verde yaw_alvo={math.degrees(self.target_yaw):.1f}°"
                )

            return

        front, left, right = self.process_scan_sectors(self.latest_scan)
        self.last_front_distance = front

        if front < self.front_obstacle_threshold:
            if right > left:
                self.target_yaw = self.normalize_angle(current_yaw - (math.pi / 2.0))
                self.transition_to_stop_then_turn(
                    self.TURN_RIGHT,
                    f"obstaculo frente={front:.2f}m esq={left:.2f}m dir={right:.2f}m -> DIREITA"
                )
            else:
                self.target_yaw = self.normalize_angle(current_yaw + (math.pi / 2.0))
                self.transition_to_stop_then_turn(
                    self.TURN_LEFT,
                    f"obstaculo frente={front:.2f}m esq={left:.2f}m dir={right:.2f}m -> ESQUERDA"
                )
            return

        self.current_state = self.GO_FORWARD
        target_speed = self.compute_forward_target_speed(front)
        self.log_transition(
            self.GO_FORWARD,
            f"caminho livre frente={front:.2f}m vel_alvo={target_speed:.2f}m/s",
        )
        self.execute_current_state()

    def execute_current_state(self) -> None:
        cmd = Twist()

        if self.current_state == self.GO_FORWARD:
            target_speed = self.compute_forward_target_speed(self.last_front_distance)
            if self.current_forward_speed < target_speed:
                self.current_forward_speed += self.acceleration_step
                self.current_forward_speed = min(self.current_forward_speed, target_speed)
            elif self.current_forward_speed > target_speed:
                self.current_forward_speed -= self.acceleration_step
                self.current_forward_speed = max(self.current_forward_speed, target_speed)
            cmd.linear.x = self.current_forward_speed
            cmd.linear.y = self.compute_lateral_command(
                self.latest_nav_cmd.linear.y, self.last_front_distance
            )
            cmd.angular.z = 0.0

        elif self.current_state == self.STOP_BEFORE_TURN:
            pass

        elif self.current_state == self.TURN_LEFT:
            cmd.angular.z = self.turn_speed

        elif self.current_state == self.TURN_RIGHT:
            cmd.angular.z = -self.turn_speed

        elif self.current_state == self.RECOVERY:
            cmd.linear.x = -0.05
            cmd.angular.z = self.turn_speed

        self.cmd_pub.publish(cmd)

    def process_scan_sectors(self, scan: LaserScan) -> Tuple[float, float, float]:
        front = self.sector_min_distance(scan, -10.0, 10.0)
        left = self.sector_min_distance(scan, 60.0, 120.0)
        right = self.sector_min_distance(scan, -120.0, -60.0)
        return front, left, right

    def sector_min_distance(self, scan: LaserScan, start_deg: float, end_deg: float) -> float:
        if not scan.ranges:
            return float("inf")

        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        n = len(scan.ranges)
        if angle_inc == 0.0:
            return float("inf")

        i0 = int((start_rad - angle_min) / angle_inc)
        i1 = int((end_rad - angle_min) / angle_inc)
        i0 = max(0, min(n - 1, i0))
        i1 = max(0, min(n - 1, i1))
        if i0 > i1:
            i0, i1 = i1, i0

        min_dist = float("inf")
        for i in range(i0, i1 + 1):
            d = scan.ranges[i]
            if math.isfinite(d):
                if scan.range_min <= d <= scan.range_max and d < min_dist:
                    min_dist = d
        return min_dist

    def is_new_color_event(self) -> bool:
        if self.current_position is None:
            return False

        if self.latest_color not in self.VALID_COLORS:
            return False

        x, y, _ = self.current_position
        for marker in self.used_markers:
            if marker["color"] != self.latest_color:
                continue
            if self.distance_2d(x, y, marker["x"], marker["y"]) <= self.marker_distance_threshold:
                return False
        return True

    def record_current_marker(self, color: str) -> None:
        if self.current_position is None:
            return
        x, y, _ = self.current_position
        self.used_markers.append({"color": color, "x": x, "y": y})
        self.get_logger().info(
            f"[MARCADOR] cor={color} pos=({x:.2f}, {y:.2f}) total={len(self.used_markers)}"
        )

    def publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_pub.publish(msg)

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def compute_forward_target_speed(self, front_distance: float) -> float:
        if not math.isfinite(front_distance):
            return self.forward_speed
        if front_distance <= self.front_obstacle_threshold:
            return 0.0
        if front_distance >= self.approach_slowdown_distance:
            return self.forward_speed

        span = self.approach_slowdown_distance - self.front_obstacle_threshold
        if span <= 1e-6:
            return self.min_approach_speed

        ratio = (front_distance - self.front_obstacle_threshold) / span
        scaled_speed = self.min_approach_speed + ratio * (
            self.forward_speed - self.min_approach_speed
        )
        return max(self.min_approach_speed, min(self.forward_speed, scaled_speed))

    def compute_lateral_command(self, nav_linear_y: float, front_distance: float) -> float:
        if not math.isfinite(front_distance):
            return nav_linear_y
        if front_distance >= self.lateral_reduce_distance:
            return nav_linear_y
        return nav_linear_y * self.lateral_reduce_factor


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MazeStateMachineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd_vel(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

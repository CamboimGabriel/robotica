#!/usr/bin/env python3

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

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
    HANDLE_COLOR = "HANDLE_COLOR"
    RECOVERY = "RECOVERY"

    VALID_COLORS = {"red", "green", "blue"}

    def __init__(self) -> None:
        super().__init__("maze_state_machine")
        
        self.set_parameters([rclpy.parameter.Parameter(
	    'use_sim_time',
	    rclpy.Parameter.Type.BOOL,
	    True
	)])

        self.forward_speed = float(self.declare_parameter("forward_speed", 0.2).value)
        self.turn_speed = float(self.declare_parameter("turn_speed", 0.6).value)
        self.front_obstacle_threshold = float(
            self.declare_parameter("front_obstacle_threshold", 0.5).value
        )
        self.side_open_threshold = float(
            self.declare_parameter("side_open_threshold", 0.6).value
        )
        self.marker_distance_threshold = float(
            self.declare_parameter("marker_distance_threshold", 0.5).value
        )
        self.loop_distance_threshold = float(
            self.declare_parameter("loop_distance_threshold", 0.2).value
        )
        self.turn_duration = float(self.declare_parameter("turn_duration", 0.9).value)
        self.recovery_duration = float(
            self.declare_parameter("recovery_duration", 1.8).value
        )
        self.handle_color_duration = float(
            self.declare_parameter("handle_color_duration", 0.2).value
        )
        self.color_turn_duration = float(
            self.declare_parameter("color_turn_duration", 0.8).value
        )
        self.u_turn_duration = float(self.declare_parameter("u_turn_duration", 1.6).value)
        self.color_red_action = str(
            self.declare_parameter("color_red_action", "left").value
        ).lower()
        self.color_green_action = str(
            self.declare_parameter("color_green_action", "right").value
        ).lower()
        self.color_blue_action = str(
            self.declare_parameter("color_blue_action", "forward").value
        ).lower()
        self.control_rate_hz = float(self.declare_parameter("control_rate_hz", 10.0).value)
        self.decision_log_interval = float(
            self.declare_parameter("decision_log_interval", 0.7).value
        )
        self.loop_history_size = int(self.declare_parameter("loop_history_size", 30).value)
        self.loop_revisit_count = int(self.declare_parameter("loop_revisit_count", 7).value)
        self.recovery_cooldown = float(
            self.declare_parameter("recovery_cooldown", 4.0).value
        )

        self.cmd_pub = self.create_publisher(Twist, "/jetauto/cmd_vel", 10) #mudei de "/cmd_vel"
        self.create_subscription(LaserScan, "/jetauto/lidar/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(String, "/color_detected", self.color_callback, 10)

        # === ADICIONEI ESTAS DUAS LINHAS AQUI ===
        self.create_subscription(Twist, "/cmd_vel_nav", self.nav_callback, 10)
        self.latest_nav_cmd = Twist()

        self.current_state = self.GO_FORWARD
        self.state_end_time = 0.0
        self.last_recovery_time = -1e9
        self.last_decision_log_time = -1e9

        self.latest_scan: Optional[LaserScan] = None
        self.current_position: Optional[Tuple[float, float, float]] = None
        self.latest_color: str = "none"

        self.position_history: Deque[Tuple[float, float]] = deque(maxlen=self.loop_history_size)
        self.used_markers: List[Dict[str, float]] = []

        self.create_timer(1.0 / self.control_rate_hz, self.control_loop)
        self.get_logger().info("Maze state machine node started.")

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.current_position = (x, y, yaw)
        self.position_history.append((x, y))

    def color_callback(self, msg: String) -> None:
        self.latest_color = msg.data.strip().lower()

    # === ADICIONEI ESTA FUNÇÃO ===
    def nav_callback(self, msg: Twist) -> None:
        self.latest_nav_cmd = msg

    def control_loop(self) -> None:
        if self.latest_scan is None or self.current_position is None:
            self.publish_cmd_vel(0.0, 0.0)
            return

        now = self.now_seconds()
        should_log = self.should_log_decision(now)
        if self.current_state in {
            self.TURN_LEFT,
            self.TURN_RIGHT,
            self.HANDLE_COLOR,
            self.RECOVERY,
        } and now < self.state_end_time:
            if should_log:
                self.get_logger().info(
                    "STATE=ACTIVE_TIMER "
                    f"active_state={self.current_state} "
                    f"remaining={max(0.0, self.state_end_time - now):.2f}s"
                )
            self.execute_current_state()
            return

        self.current_state = self.GO_FORWARD

        if self.is_looping() and (now - self.last_recovery_time) > self.recovery_cooldown:
            if should_log:
                self.get_logger().info(
                    "STATE_DECISION reason=loop_detected "
                    f"recovery_cooldown={self.recovery_cooldown:.2f}s "
                    f"time_since_last_recovery={now - self.last_recovery_time:.2f}s "
                    f"action={self.RECOVERY}"
                )
            self.transition_to(self.RECOVERY, self.recovery_duration)
            self.last_recovery_time = now
            self.execute_current_state()
            return

        if self.is_new_color_event():
            if should_log:
                self.get_logger().info(
                    "STATE_DECISION reason=color_event "
                    f"color={self.latest_color} action=color_mapping"
                )
            self.record_current_marker(self.latest_color)
            self.transition_to_color_action(self.latest_color)
            self.execute_current_state()
            return

        front, left, right = self.process_scan_sectors(self.latest_scan)
        if should_log:
            self.get_logger().info(
                "STATE_SENSORS "
                f"front={self.format_distance(front)} "
                f"left={self.format_distance(left)} "
                f"right={self.format_distance(right)} "
                f"front_threshold={self.front_obstacle_threshold:.3f} "
                f"side_threshold={self.side_open_threshold:.3f}"
            )
        if front < self.front_obstacle_threshold:
            turn_state = self.choose_turn_direction(left, right)
            if should_log:
                self.get_logger().info(
                    "STATE_DECISION reason=front_blocked "
                    f"left={self.format_distance(left)} "
                    f"right={self.format_distance(right)} "
                    f"action={turn_state}"
                )
            self.transition_to(turn_state, self.turn_duration)
            self.execute_current_state()
            return

        if should_log:
            self.get_logger().info(
                f"STATE_DECISION reason=path_clear action={self.GO_FORWARD}"
            )
        self.current_state = self.GO_FORWARD
        self.execute_current_state()

    def process_scan_sectors(self, scan: LaserScan) -> Tuple[float, float, float]:
        front = self.sector_min_distance(scan, -15.0, 15.0)
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

    def is_looping(self) -> bool:
        if len(self.position_history) < self.loop_history_size:
            return False
        if self.current_position is None:
            return False

        x, y, _ = self.current_position
        close_count = 0
        for px, py in self.position_history:
            if self.distance_2d(x, y, px, py) <= self.loop_distance_threshold:
                close_count += 1
        return close_count >= self.loop_revisit_count

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
            f"New marker: color={color}, x={x:.2f}, y={y:.2f}, total={len(self.used_markers)}"
        )

    def transition_to_color_action(self, color: str) -> None:
        action_map = {
            "red": self.color_red_action,
            "green": self.color_green_action,
            "blue": self.color_blue_action,
        }
        action = action_map.get(color, "forward")

        if action == "left":
            self.transition_to(self.TURN_LEFT, self.color_turn_duration)
        elif action == "right":
            self.transition_to(self.TURN_RIGHT, self.color_turn_duration)
        elif action == "u_turn":
            self.transition_to(self.RECOVERY, self.u_turn_duration)
        else:
            self.transition_to(self.HANDLE_COLOR, self.handle_color_duration)

        self.get_logger().info(f"Color action: color={color}, action={action}")

    def transition_to(self, state: str, duration: float = 0.0) -> None:
        self.current_state = state
        self.state_end_time = self.now_seconds() + max(0.0, duration)

    def execute_current_state(self) -> None:
        self.get_logger().info(f"STATE: {self.current_state}")
        if self.current_state == self.GO_FORWARD:
            self.get_logger().info("ENTROU NO GO_FORWARD")
            cmd = Twist()
            cmd.linear.x = self.latest_nav_cmd.linear.x
            cmd.linear.y = self.latest_nav_cmd.linear.y
            cmd.angular.z = self.latest_nav_cmd.angular.z
            self.get_logger().info(
                f"CMD NAV: x={cmd.linear.x}, y={cmd.linear.y}, z={cmd.angular.z}"
            )
            self.cmd_pub.publish(cmd)

        elif self.current_state == self.TURN_LEFT:
            self.publish_cmd_vel(0.0, self.turn_speed)

        elif self.current_state == self.TURN_RIGHT:
            self.publish_cmd_vel(0.0, -self.turn_speed)

        elif self.current_state == self.HANDLE_COLOR:
            self.publish_cmd_vel(self.forward_speed * 0.6, 0.0)

        elif self.current_state == self.RECOVERY:
            self.publish_cmd_vel(0.0, self.turn_speed)

        else:
            self.publish_cmd_vel(0.0, 0.0)

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

    def should_log_decision(self, now: float) -> bool:
        if (now - self.last_decision_log_time) >= self.decision_log_interval:
            self.last_decision_log_time = now
            return True
        return False

    def choose_turn_direction(self, left: float, right: float) -> str:
        if math.isinf(left) and math.isinf(right):
            return self.TURN_LEFT
        if math.isinf(right):
            return self.TURN_RIGHT
        if math.isinf(left):
            return self.TURN_LEFT
        if right > left:
            return self.TURN_RIGHT
        return self.TURN_LEFT

    @staticmethod
    def format_distance(value: float) -> str:
        if math.isinf(value):
            return "inf"
        return f"{value:.3f}"


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


if __name__ == "__main__":
    main()

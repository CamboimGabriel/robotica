#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
import math


class MazeRobot(Node):

    def __init__(self):
        super().__init__('maze_robot')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.color_sub = self.create_subscription(
            String, '/color_detected',
            self.color_callback,
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        # Estado
        self.state = "FORWARD"
        self.latest_scan = None
        self.current_yaw = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_color = "none"
        self.heading_target = None

        # Curva
        self.curve_direction = 0
        self.curve_strength = 0.5
        self.curve_forward_speed = 0.08

        # Distâncias
        self.front_slow_threshold = 1.0
        self.front_stop_threshold = 0.65
        self.front_emergency_threshold = 0.40

        # Velocidade
        self.linear_speed = 0.15
        self.min_linear_speed = 0.04

        # Heading
        self.heading_kp = 1.2
        self.heading_max_correction = 0.18
        self.heading_tolerance = 0.01

        # Anti-loop
        self.forward_grace_sec = 1.0
        self.forward_grace_until = 0.0
        self.wall_recount_radius = 0.85
        self.counted_walls = {
            "red": [],
            "green": [],
            "blue": [],
        }

        # 🔥 FILTRO LIDAR
        self.front_history = []
        self.history_size = 5

        self.get_logger().info("🚀 MazeRobot FINAL ROBUSTO iniciado")

    # =========================

    def scan_callback(self, msg):
        self.latest_scan = msg

    def odom_callback(self, msg):
        o = msg.pose.pose.orientation
        p = msg.pose.pose.position
        _, _, yaw = self.euler_from_quaternion(o.x, o.y, o.z, o.w)
        self.current_yaw = yaw
        self.current_x = p.x
        self.current_y = p.y

    def color_callback(self, msg):
        self.current_color = msg.data

    # =========================

    def control_loop(self):
        if self.latest_scan is None:
            return

        now = self.now_sec()

        # 🔥 DISTÂNCIA FILTRADA
        front_raw = self.sector_filtered_distance(self.latest_scan, -25.0, 25.0)

        self.front_history.append(front_raw)
        if len(self.front_history) > self.history_size:
            self.front_history.pop(0)

        front = sum(self.front_history) / len(self.front_history)

        left = self.sector_mean_distance(self.latest_scan, 60.0, 100.0)
        right = self.sector_mean_distance(self.latest_scan, -100.0, -60.0)

        cmd = Twist()

        # 🚨 EMERGÊNCIA
        if front < self.front_emergency_threshold and front > 0.10:
            self.get_logger().warn(f"🚨 EMERGÊNCIA {front:.2f}m")

            cmd.linear.x = 0.0
            cmd.angular.z = -0.8 if right > left else 0.8
            self.cmd_pub.publish(cmd)
            return

        # COR
        if self.state == "FORWARD" and now >= self.forward_grace_until:

            if self.current_color == "green":
                if not self.should_count_wall("green", front):
                    return
                self.get_logger().info("🟢 curva direita")
                self.start_curve(-1)
                return

            elif self.current_color == "red":
                if not self.should_count_wall("red", front):
                    return
                self.get_logger().info("🔴 curva esquerda")
                self.start_curve(+1)
                return

        # OBSTÁCULO
        obstacle = front < self.front_stop_threshold and front > 0.10

        if self.state == "FORWARD":

            if self.heading_target is None:
                self.heading_target = self.current_yaw

            heading_error = self.normalize_angle(
                self.heading_target - self.current_yaw)

            if abs(heading_error) > self.heading_tolerance:
                cmd.angular.z = self.clamp(
                    self.heading_kp * heading_error,
                    -self.heading_max_correction,
                    self.heading_max_correction
                )

            # desaceleração forte
            if front < self.front_slow_threshold:
                ratio = (front - self.front_stop_threshold) / (
                    self.front_slow_threshold - self.front_stop_threshold)
                ratio = max(0.0, min(1.0, ratio))
                ratio = ratio ** 2.5

                cmd.linear.x = max(self.min_linear_speed,
                                   self.linear_speed * ratio)
            else:
                cmd.linear.x = self.linear_speed

            if obstacle:
                self.get_logger().info(f"🚧 obstáculo {front:.2f}m")
                if right > left:
                    self.start_curve(-1)
                else:
                    self.start_curve(+1)

        elif self.state == "CURVE":

            cmd.linear.x = self.curve_forward_speed

            dynamic = self.curve_strength * (1.0 + 1.5 / max(front, 0.3))
            cmd.angular.z = self.curve_direction * dynamic

            if front > self.front_slow_threshold:
                self.get_logger().info("↪️ fim curva")
                self.state = "FORWARD"
                self.heading_target = self.current_yaw
                self.forward_grace_until = self.now_sec() + self.forward_grace_sec

        self.cmd_pub.publish(cmd)

    # =========================

    def start_curve(self, direction):
        self.state = "CURVE"
        self.curve_direction = direction
        self.current_color = "none"

    def should_count_wall(self, color, wall_distance):
        if color not in self.counted_walls:
            return True

        wall_x, wall_y = self.estimate_wall_position(wall_distance)

        for wx, wy in self.counted_walls[color]:
            d = math.hypot(wall_x - wx, wall_y - wy)
            if d <= self.wall_recount_radius:
                self.get_logger().info(
                    f"↩️ {color} já contado (d={d:.2f}m)"
                )
                self.current_color = "none"
                return False

        self.counted_walls[color].append((wall_x, wall_y))
        self.get_logger().info(
            f"✅ {color} novo muro salvo em ({wall_x:.2f}, {wall_y:.2f})"
        )
        return True

    def estimate_wall_position(self, wall_distance):
        if not math.isfinite(wall_distance) or wall_distance <= 0.10:
            return self.current_x, self.current_y

        wx = self.current_x + wall_distance * math.cos(self.current_yaw)
        wy = self.current_y + wall_distance * math.sin(self.current_yaw)
        return wx, wy

    # =========================
    # 🔥 FILTRO MEDIANA
    # =========================

    def sector_filtered_distance(self, scan, start_deg, end_deg):
        values = []

        i0 = int((math.radians(start_deg) - scan.angle_min) / scan.angle_increment)
        i1 = int((math.radians(end_deg) - scan.angle_min) / scan.angle_increment)

        i0 = max(0, min(len(scan.ranges)-1, i0))
        i1 = max(0, min(len(scan.ranges)-1, i1))

        if i0 > i1:
            i0, i1 = i1, i0

        for i in range(i0, i1+1):
            d = scan.ranges[i]
            if math.isfinite(d) and 0.05 < d < 5.0:
                values.append(d)

        if not values:
            return float("inf")

        values.sort()
        return values[len(values)//2]

    def sector_mean_distance(self, scan, start_deg, end_deg):
        vals = []

        i0 = int((math.radians(start_deg) - scan.angle_min) / scan.angle_increment)
        i1 = int((math.radians(end_deg) - scan.angle_min) / scan.angle_increment)

        i0 = max(0, min(len(scan.ranges)-1, i0))
        i1 = max(0, min(len(scan.ranges)-1, i1))

        if i0 > i1:
            i0, i1 = i1, i0

        for i in range(i0, i1+1):
            d = scan.ranges[i]
            if math.isfinite(d):
                vals.append(d)

        return sum(vals)/len(vals) if vals else float("inf")

    # =========================

    def clamp(self, v, mn, mx):
        return max(mn, min(v, mx))

    def normalize_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))

    def euler_from_quaternion(self, x, y, z, w):
        t0 = 2*(w*z + x*y)
        t1 = 1 - 2*(y*y + z*z)
        yaw = math.atan2(t0, t1)
        return 0, 0, yaw

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9


def main():
    rclpy.init()
    node = MazeRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
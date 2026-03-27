#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math


class OmniNavLayer(Node):
    def __init__(self):
        super().__init__('omni_nav_layer')

        #self.declare_parameter('use_sim_time', True)

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        self.subscription = self.create_subscription(
            LaserScan,
            '/jetauto/lidar/scan',
            self.lidar_callback,
            10
        )

        self.dist_seguranca_frontal = 0.7
        self.dist_parede_alvo = 0.45

        self.vel_cruzeiro = 0.5
        self.k_lateral = 1.0

        self.max_linear_x = 0.5
        self.max_linear_y = 0.3
        self.max_angular_z = 1.0

        self.get_logger().info("OmniNavLayer iniciado")

    def lidar_callback(self, msg):
        frente = self.sector_min_distance(msg, -10.0, 10.0)
        direita = self.sector_min_distance(msg, -100.0, -70.0)
        frente_dir = self.sector_min_distance(msg, -60.0, -20.0)

        self.gerar_comando(frente, frente_dir, direita)

    def sector_min_distance(self, scan, start_deg, end_deg):
        if not scan.ranges or scan.angle_increment == 0.0:
            return float('inf')

        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)

        i0 = int((start_rad - scan.angle_min) / scan.angle_increment)
        i1 = int((end_rad - scan.angle_min) / scan.angle_increment)

        n = len(scan.ranges)
        i0 = max(0, min(n - 1, i0))
        i1 = max(0, min(n - 1, i1))

        if i0 > i1:
            i0, i1 = i1, i0

        best = float('inf')
        for i in range(i0, i1 + 1):
            d = scan.ranges[i]
            if math.isfinite(d) and scan.range_min <= d <= scan.range_max:
                if d < best:
                    best = d

        return best

    def gerar_comando(self, dist_frente, dist_frente_dir, dist_dir):
        msg = Twist()

        if dist_frente < self.dist_seguranca_frontal:
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.angular.z = 0.8

        elif dist_frente_dir < 0.5:
            msg.linear.x = 0.2
            msg.linear.y = 0.0
            msg.angular.z = 0.5

        else:
            msg.linear.x = self.vel_cruzeiro

            erro_lateral = dist_dir - self.dist_parede_alvo
            msg.linear.y = -self.k_lateral * erro_lateral

            msg.linear.x = min(msg.linear.x, self.max_linear_x)
            msg.linear.y = max(min(msg.linear.y, self.max_linear_y), -self.max_linear_y)
            msg.angular.z = max(min(msg.angular.z, self.max_angular_z), -self.max_angular_z)

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OmniNavLayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

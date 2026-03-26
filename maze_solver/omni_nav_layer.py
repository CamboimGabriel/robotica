#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math


class OmniNavLayer(Node):
    def __init__(self):
        super().__init__('omni_nav_layer')
        
        self.set_parameters([rclpy.parameter.Parameter(
	    'use_sim_time',
	    rclpy.Parameter.Type.BOOL,
	    True
	)])
        
        # Publica para a state machine consumir
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        
        self.subscription = self.create_subscription(
            LaserScan, 
            '/jetauto/lidar/scan', 
            self.lidar_callback, 
            10)
            
        self.dist_seguranca_frontal = 0.4
        self.dist_parede_alvo = 0.35
        
        self.vel_cruzeiro = 0.25
        self.k_lateral = 0.8
        
        self.max_linear_x = 0.3
        self.max_linear_y = 0.2
        self.max_angular_z = 0.8

        self.get_logger().info("OmniNavLayer iniciado")

    def lidar_callback(self, msg):
        ranges = [
            x if msg.range_min < x < msg.range_max else 10.0
            for x in msg.ranges
        ]

        n = len(ranges)

        frente = min(ranges[int(n*0.48): int(n*0.52)])
        esquerda = min(ranges[int(n*0.70): int(n*0.80)])
        direita = min(ranges[int(n*0.20): int(n*0.30)])
        frente_dir = min(ranges[int(n*0.30): int(n*0.40)])
        frente_esq = min(ranges[int(n*0.60): int(n*0.70)])

        self.gerar_comando(frente, frente_dir, frente_esq, direita, esquerda)

    def gerar_comando(self, dist_frente, dist_frente_dir, dist_frente_esq, dist_dir, dist_esq):
        msg = Twist()
        largura_corredor = dist_dir + dist_esq

        if dist_frente < self.dist_seguranca_frontal:
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.angular.z = 0.8

        elif dist_frente_dir < 0.4:
            msg.linear.x = 0.15
            msg.linear.y = 0.1
            msg.angular.z = 0.4

        elif dist_frente_esq < 0.4:
            msg.linear.x = 0.15
            msg.linear.y = -0.1
            msg.angular.z = -0.4

        else:
            # In narrow corridors, center between both walls
            if largura_corredor < 1.0:
                erro_lateral = (dist_esq - dist_dir) / 2.0
                k = self.k_lateral * 0.5
                vel = self.vel_cruzeiro * 0.5
            else:
                erro_lateral = dist_dir - self.dist_parede_alvo
                k = self.k_lateral
                vel = self.vel_cruzeiro

            msg.linear.x = min(vel, self.max_linear_x)
            msg.linear.y = -k * erro_lateral
            msg.linear.y = max(min(msg.linear.y, self.max_linear_y), -self.max_linear_y)
            msg.angular.z = 0.0

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OmniNavLayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

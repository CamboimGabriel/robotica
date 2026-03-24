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
            
        self.dist_seguranca_frontal = 0.7   
        self.dist_parede_alvo = 0.45
        
        self.vel_cruzeiro = 0.5             
        self.k_lateral = 1.0                
        
        # limites 
        self.max_linear_x = 0.5
        self.max_linear_y = 0.3
        self.max_angular_z = 1.0

        self.get_logger().info("OmniNavLayer iniciado")

    def lidar_callback(self, msg):
        
        ranges = [
            x if msg.range_min < x < msg.range_max else 10.0
            for x in msg.ranges
        ]

        n = len(ranges)

        frente = min(ranges[int(n*0.48): int(n*0.52)])
        direita = min(ranges[int(n*0.20): int(n*0.30)])
        frente_dir = min(ranges[int(n*0.30): int(n*0.40)])  # NOVO (quina)

        self.gerar_comando(frente, frente_dir, direita)

    def gerar_comando(self, dist_frente, dist_frente_dir, dist_dir):
        msg = Twist()

        # evitar colisão frontal
        if dist_frente < self.dist_seguranca_frontal:
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.angular.z = 0.8  # giro forte

        # evitar bater em quina direita
        elif dist_frente_dir < 0.5:
            msg.linear.x = 0.2
            msg.linear.y = 0.0
            msg.angular.z = 0.5

        else:
            # movimento principal
            msg.linear.x = self.vel_cruzeiro

            # controle lateral (
            erro_lateral = dist_dir - self.dist_parede_alvo
            msg.linear.y = -self.k_lateral * erro_lateral

            # limitadores
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

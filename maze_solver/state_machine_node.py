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
    RECOVERY = "RECOVERY"
    STOP_BEFORE_TURN = "STOP_BEFORE_TURN"  # NOVO ESTADO DE PARADA

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
        self.marker_distance_threshold = float(
            self.declare_parameter("marker_distance_threshold", 0.5).value
        )
        self.control_rate_hz = float(self.declare_parameter("control_rate_hz", 10.0).value)

        # === VARIÁVEIS PARA GIRO EXATO, PARADA E ACELERAÇÃO ===
        self.target_yaw: Optional[float] = None
        self.yaw_tolerance = 0.05  # Tolerância em radianos
        
        self.next_state_after_stop = ""
        self.stop_wait_duration = 0.4
        self.current_forward_speed = 0.0
        self.acceleration_step = 0.02

        # Publishers e Subscribers
        self.cmd_pub = self.create_publisher(Twist, "/jetauto/cmd_vel", 10)
        self.create_subscription(LaserScan, "/jetauto/lidar/scan", self.scan_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(String, "/color_detected", self.color_callback, 10)
        self.create_subscription(Twist, "/cmd_vel_nav", self.nav_callback, 10)

        self.latest_nav_cmd = Twist()
        self.current_state = self.GO_FORWARD
        self.state_end_time = 0.0
        
        self.latest_scan: Optional[LaserScan] = None
        self.current_position: Optional[Tuple[float, float, float]] = None
        self.latest_color: str = "none"
        self.used_markers: List[Dict[str, float]] = []

        self.create_timer(1.0 / self.control_rate_hz, self.control_loop)
        self.get_logger().info("Maze state machine node started. Prontos para o labirinto!")

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

    def normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def transition_to_stop_then_turn(self, next_state: str) -> None:
        """Trava o robô e aguarda antes de iniciar a curva."""
        self.current_state = self.STOP_BEFORE_TURN
        self.next_state_after_stop = next_state
        self.state_end_time = self.now_seconds() + self.stop_wait_duration
        self.current_forward_speed = 0.0  # Zera a rampa de aceleração
        self.publish_cmd_vel(0.0, 0.0)    # Freia imediatamente
        self.get_logger().info(f"Freando! Esperando {self.stop_wait_duration}s antes de girar.")

    def control_loop(self) -> None:
        if self.latest_scan is None or self.current_position is None:
            self.publish_cmd_vel(0.0, 0.0)
            return

        x, y, current_yaw = self.current_position
        now = self.now_seconds()

        # 0. GERENCIA O ESTADO DE PARADA TOTAL (INÉRCIA)
        if self.current_state == self.STOP_BEFORE_TURN:
            if now >= self.state_end_time:
                self.current_state = self.next_state_after_stop
                self.get_logger().info(f"Parada concluída. Iniciando {self.next_state_after_stop}.")
            else:
                self.publish_cmd_vel(0.0, 0.0)
            return

        # 1. VERIFICA SE ESTÁ NO MEIO DE UMA CURVA (ODOMETRIA)
        if self.current_state in {self.TURN_LEFT, self.TURN_RIGHT}:
            if self.target_yaw is not None:
                angle_diff = abs(self.normalize_angle(self.target_yaw - current_yaw))
                
                if angle_diff <= self.yaw_tolerance:
                    self.get_logger().info("Curva concluída! Retomando aceleração em linha reta.")
                    self.target_yaw = None
                    self.current_state = self.GO_FORWARD
                    self.current_forward_speed = 0.0 # Garante que acelera do zero
                    self.latest_color = "none"       # Limpa a cor para não ler novamente a mesma parede
                else:
                    self.execute_current_state()
                    return

        # 2. VERIFICA SE ENCONTROU UMA COR NOVA
        if self.is_new_color_event():
            self.record_current_marker(self.latest_color)
            
            if self.latest_color == "red":
                self.get_logger().info("Parede Vermelha! Preparando giro para a ESQUERDA.")
                self.target_yaw = self.normalize_angle(current_yaw + (math.pi / 2.0))
                self.transition_to_stop_then_turn(self.TURN_LEFT)
            
            elif self.latest_color == "green":
                self.get_logger().info("Parede Verde! Preparando giro para a DIREITA.")
                self.target_yaw = self.normalize_angle(current_yaw - (math.pi / 2.0))
                self.transition_to_stop_then_turn(self.TURN_RIGHT)
            
            return

        # 3. VERIFICA OBSTÁCULOS COM O LIDAR (A PAREDE PRETA/CRUZAMENTO T)
        front, left, right = self.process_scan_sectors(self.latest_scan)
        
        if front < self.front_obstacle_threshold:
            self.get_logger().info(f"Obstáculo frontal a {front:.2f}m. Decidindo lado...")
            
            if right > left:
                self.get_logger().info("Caminho livre à DIREITA.")
                self.target_yaw = self.normalize_angle(current_yaw - (math.pi / 2.0))
                self.transition_to_stop_then_turn(self.TURN_RIGHT)
            else:
                self.get_logger().info("Caminho livre à ESQUERDA.")
                self.target_yaw = self.normalize_angle(current_yaw + (math.pi / 2.0))
                self.transition_to_stop_then_turn(self.TURN_LEFT)
            
            return

        # Se não tem obstáculo nem cor, continua a lógica de seguir em frente
        self.current_state = self.GO_FORWARD
        self.execute_current_state()

    def execute_current_state(self) -> None:
        """
        Executa a lógica de movimentação baseada no estado atual.
        Agora com suporte total a movimentos omnidirecionais (strafing).
        """
        self.get_logger().info(f"ESTADO ATUAL: {self.current_state}")
        
        cmd = Twist()

        # --- ESTADO: SEGUIR EM FRENTE ---
        if self.current_state == self.GO_FORWARD:
            # 1. Rampa de Aceleração Suave
            if self.current_forward_speed < self.forward_speed:
                self.current_forward_speed += self.acceleration_step
                self.current_forward_speed = min(self.current_forward_speed, self.forward_speed)

            # 2. Movimento para frente (Eixo X) controlado pela State Machine
            cmd.linear.x = self.current_forward_speed 
            
            # 3. Centralização Lateral (Eixo Y) vinda do OmniNavLayer
            # O robô desliza como um caranguejo para se manter no centro, 
            # mantendo o nariz (angular.z) sempre em ZERO (apontado para frente).
            cmd.linear.y = self.latest_nav_cmd.linear.y
            cmd.angular.z = 0.0 
            
            self.get_logger().info(f"AVANÇANDO: x={cmd.linear.x:.2f}, y={cmd.linear.y:.2f}")

        # --- ESTADO: PARADA E SEGURANÇA ---
        elif self.current_state == self.STOP_BEFORE_TURN:
            # Garante imobilidade total antes de iniciar uma rotação
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("FREANDO: Aguardando inércia parar...")

        # --- ESTADOS: ROTAÇÃO DE 90 GRAUS ---
        elif self.current_state == self.TURN_LEFT:
            # Gira no próprio eixo (sem sair do lugar)
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = self.turn_speed
            self.get_logger().info("GIRANDO: Esquerda (90°)")

        elif self.current_state == self.TURN_RIGHT:
            # Gira no próprio eixo (sem sair do lugar)
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = -self.turn_speed
            self.get_logger().info("GIRANDO: Direita (90°)")

        # --- ESTADO: RECUPERAÇÃO / LOOP ---
        elif self.current_state == self.RECOVERY:
            # Dá uma leve ré e gira para tentar achar um caminho novo
            cmd.linear.x = -0.05
            cmd.angular.z = self.turn_speed
            self.get_logger().warn("RECUPERAÇÃO: Saindo de possível colisão/loop.")

        # --- SEGURANÇA PADRÃO ---
        else:
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0

        # Publica o comando final para os motores
        self.cmd_pub.publish(cmd)

    def process_scan_sectors(self, scan: LaserScan) -> Tuple[float, float, float]:
        front = self.sector_min_distance(scan, -10.0, 10.0)
        
        # A visão lateral continua ampla para ele medir bem as distâncias
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

#!/usr/bin/env python3
"""
Nodo: stateBot.py
Paquete: avlab7
Descripción: Nodo para definir, calcular y publicar los estados del TurtleBot3
             tanto para aprendizaje por refuerzo tabular (Q-Learning) como continuo (PPO).
             Cumple con la Actividad 1.2 del Laboratorio 7.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from std_msgs.msg import Int32, Float32MultiArray, String


def euler_from_quaternion(x, y, z, w):
    """Convierte un cuaternión (x, y, z, w) a ángulo de Euler yaw (rotación en Z)."""
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


class StateBot(Node):
    """
    Nodo de ROS 2 que calcula el estado actual del robot combinando:
    - Odometría (/odom): Posición (x, y) y orientación (yaw).
    - Sensor LiDAR (/scan): Distancias a obstáculos en 5 sectores clave.
    - Meta objetivo (/goal_pose): Distancia euclidiana y ángulo relativo hacia el objetivo.
    """

    def __init__(self):
        super().__init__('state_bot')

        # Parámetros del entorno
        self.declare_parameter('goal_x', 1.5)
        self.declare_parameter('goal_y', 1.5)
        self.declare_parameter('world_min', -2.0)
        self.declare_parameter('world_max', 2.0)
        self.declare_parameter('grid_size', 4)       # Cuadrícula 4x4 = 16 estados (como FrozenLake)
        self.declare_parameter('print_rate', 1.0)     # Frecuencia de impresión en Hz

        self.goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        self.goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        self.world_min = self.get_parameter('world_min').get_parameter_value().double_value
        self.world_max = self.get_parameter('world_max').get_parameter_value().double_value
        self.grid_size = self.get_parameter('grid_size').get_parameter_value().integer_value
        self.print_rate = self.get_parameter('print_rate').get_parameter_value().double_value

        # Estado del robot
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.has_odom = False

        # Sectores LiDAR (distancias en metros)
        self.laser_front = 3.5
        self.laser_front_left = 3.5
        self.laser_front_right = 3.5
        self.laser_left = 3.5
        self.laser_right = 3.5
        self.min_laser_distance = 3.5
        self.has_scan = False

        # Publicadores
        self.state_pub = self.create_publisher(Int32, '/state', 10)
        self.state_vector_pub = self.create_publisher(Float32MultiArray, '/state_vector', 10)
        self.state_info_pub = self.create_publisher(String, '/state_info', 10)

        # Suscriptores
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.goal_sub = self.create_subscription(Point, '/goal_pose', self.goal_callback, 10)

        # Timer para cálculo e impresión periódica
        self.timer = self.create_timer(1.0 / self.print_rate, self.state_loop)

        self.get_logger().info('========================================================')
        self.get_logger().info('Nodo stateBot inicializado.')
        self.get_logger().info(f'Meta inicial: ({self.goal_x:.2f}, {self.goal_y:.2f})')
        self.get_logger().info(f'Cuadrícula discreta: {self.grid_size}x{self.grid_size} = {self.grid_size*self.grid_size} estados')
        self.get_logger().info('Esperando datos de /odom y /scan...')
        self.get_logger().info('========================================================')

    def odom_callback(self, msg: Odometry):
        """Lee la posición y orientación del robot."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.has_odom = True

    def scan_callback(self, msg: LaserScan):
        """Procesa los rayos del escáner láser dividiéndolos en sectores clave."""
        ranges = msg.ranges
        num_ranges = len(ranges)
        if num_ranges == 0:
            return

        def safe_sector_min(start_deg, end_deg):
            """Calcula la distancia mínima en un sector angular."""
            # Mapear grados [-180, 180] o [0, 360] a índices del array
            idx_start = int((start_deg % 360) / 360.0 * num_ranges)
            idx_end = int((end_deg % 360) / 360.0 * num_ranges)
            
            if idx_start <= idx_end:
                sector = ranges[idx_start:idx_end+1]
            else:
                sector = ranges[idx_start:] + ranges[:idx_end+1]
            
            valid = [r for r in sector if msg.range_min < r < msg.range_max]
            return min(valid) if len(valid) > 0 else msg.range_max

        # Sectores frontales y laterales
        self.laser_front = min(safe_sector_min(340, 360), safe_sector_min(0, 20))
        self.laser_front_left = safe_sector_min(20, 70)
        self.laser_left = safe_sector_min(70, 110)
        self.laser_front_right = safe_sector_min(290, 340)
        self.laser_right = safe_sector_min(250, 290)

        all_valid = [r for r in ranges if msg.range_min < r < msg.range_max]
        self.min_laser_distance = min(all_valid) if len(all_valid) > 0 else msg.range_max
        self.has_scan = True

    def goal_callback(self, msg: Point):
        """Actualiza la posición de la meta."""
        self.goal_x = msg.x
        self.goal_y = msg.y
        self.get_logger().info(f'Nueva meta recibida en /goal_pose: ({self.goal_x:.2f}, {self.goal_y:.2f})')

    def compute_discrete_grid_state(self) -> int:
        """
        Calcula el estado discreto basado en una cuadrícula (NxN).
        Divide el plano [world_min, world_max] en celdas regulares.
        Ej: para 4x4 (16 celdas), los IDs van de 0 a 15 (análogo a FrozenLake).
        """
        span = self.world_max - self.world_min
        step = span / float(self.grid_size)

        col = int((self.robot_x - self.world_min) / step)
        row = int((self.robot_y - self.world_min) / step)

        col = max(0, min(col, self.grid_size - 1))
        row = max(0, min(row, self.grid_size - 1))

        # ID de estado: fila * N + col
        return row * self.grid_size + col

    def compute_goal_metrics(self):
        """Calcula la distancia euclidiana y el ángulo relativo a la meta."""
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        distance = math.sqrt(dx * dx + dy * dy)

        angle_to_goal = math.atan2(dy, dx)
        angle_diff = angle_to_goal - self.robot_yaw

        # Normalizar a [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2.0 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2.0 * math.pi

        return distance, angle_diff

    def state_loop(self):
        """Cálculo, publicación e impresión del estado."""
        if not self.has_odom:
            self.get_logger().info('Esperando primer mensaje de /odom...')
            return

        # Métricas hacia la meta
        distance_to_goal, angle_to_goal = self.compute_goal_metrics()

        # Estado discreto en grilla
        discrete_state = self.compute_discrete_grid_state()

        # Vector continuo para RL (PPO)
        # [dist_goal_norm, angle_goal_norm, laser_f, laser_fl, laser_fr, laser_l, laser_r]
        continuous_vector = [
            float(min(distance_to_goal / 5.0, 1.0)),
            float(angle_to_goal / math.pi),
            float(min(self.laser_front / 3.5, 1.0)),
            float(min(self.laser_front_left / 3.5, 1.0)),
            float(min(self.laser_front_right / 3.5, 1.0)),
            float(min(self.laser_left / 3.5, 1.0)),
            float(min(self.laser_right / 3.5, 1.0)),
        ]

        # Publicar estado discreto
        state_msg = Int32()
        state_msg.data = discrete_state
        self.state_pub.publish(state_msg)

        # Publicar vector continuo
        vec_msg = Float32MultiArray()
        vec_msg.data = continuous_vector
        self.state_vector_pub.publish(vec_msg)

        # Publicar info
        col = discrete_state % self.grid_size
        row = discrete_state // self.grid_size
        info_text = (
            f'State={discrete_state} (Fila={row}, Col={col}) | '
            f'Pos=({self.robot_x:.2f}, {self.robot_y:.2f}) | '
            f'DistMeta={distance_to_goal:.2f}m | '
            f'AngMeta={math.degrees(angle_to_goal):.1f}° | '
            f'MinLaser={self.min_laser_distance:.2f}m'
        )
        info_msg = String()
        info_msg.data = info_text
        self.state_info_pub.publish(info_msg)

        # Impresión formateada del estado
        self.get_logger().info('------------------ ESTADO ACTUAL DEL ROBOT ------------------')
        self.get_logger().info(f'• Estado Discreto Grid [{self.grid_size}x{self.grid_size}]: ID = {discrete_state} (Fila: {row}, Columna: {col})')
        self.get_logger().info(f'• Posición Robot: x = {self.robot_x:6.2f} m, y = {self.robot_y:6.2f} m | Yaw = {math.degrees(self.robot_yaw):6.1f}°')
        self.get_logger().info(f'• Meta Objetivo:  x = {self.goal_x:6.2f} m, y = {self.goal_y:6.2f} m')
        self.get_logger().info(f'• Relativo Meta:  Distancia = {distance_to_goal:5.2f} m | Ángulo relativo = {math.degrees(angle_to_goal):6.1f}°')
        self.get_logger().info(
            f'• Sensor Láser:   Frente={self.laser_front:.2f}m, Izq={self.laser_left:.2f}m, '
            f'Der={self.laser_right:.2f}m | Mínimo={self.min_laser_distance:.2f}m'
        )
        self.get_logger().info('-------------------------------------------------------------')


def main(args=None):
    rclpy.init(args=args)
    state_bot = StateBot()

    try:
        rclpy.spin(state_bot)
    except KeyboardInterrupt:
        state_bot.get_logger().info('Deteniendo stateBot...')
    finally:
        state_bot.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

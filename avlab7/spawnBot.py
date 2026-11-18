#!/usr/bin/env python3
"""
Nodo: spawnBot.py
Paquete: avlab7
Descripción: Nodo para posicionar el robot TurtleBot3 en una posición aleatoria
             válida dentro de las zonas libres delimitadas de la simulación en Gazebo,
             asegurando que no aparezca debajo ni encima de ningún obstáculo.
             Cumple con la Actividad 1.4 del Laboratorio 7.
"""

import sys
import math
import random
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose2D, Twist
from std_srvs.srv import Trigger


class SpawnBot(Node):
    """
    Nodo de ROS 2 que delimita las zonas libres del mundo de Gazebo (turtlebot3_world)
    y reposiciona aleatoriamente al robot en una zona segura mediante el servicio
    /set_entity_state de gazebo_msgs.
    """

    # Delimitación de las ZONAS LIBRES en turtlebot3_world:
    # Coordenadas rectangulares [x_min, x_max, y_min, y_max] donde no existen obstáculos.
    FREE_ZONES = [
        # Zona 1: Franja exterior Oeste (Spawn clásico)
        {'name': 'Oeste',    'bounds': [-2.0, -1.4, -1.8, 1.8]},
        # Zona 2: Franja exterior Este
        {'name': 'Este',     'bounds': [ 1.4,  2.0, -1.8, 1.8]},
        # Zona 3: Franja exterior Norte
        {'name': 'Norte',    'bounds': [-1.4,  1.4,  1.4, 2.0]},
        # Zona 4: Franja exterior Sur
        {'name': 'Sur',      'bounds': [-1.4,  1.4, -2.0, -1.4]},
        # Zona 5: Pasillo interior Noroeste
        {'name': 'Pasillo_NO', 'bounds': [-0.6, -0.2,  0.4, 1.0]},
        # Zona 6: Pasillo interior Noreste
        {'name': 'Pasillo_NE', 'bounds': [ 0.2,  0.6,  0.4, 1.0]},
        # Zona 7: Pasillo interior Suroeste
        {'name': 'Pasillo_SO', 'bounds': [-0.6, -0.2, -1.0, -0.4]},
        # Zona 8: Pasillo interior Sureste
        {'name': 'Pasillo_SE', 'bounds': [ 0.2,  0.6, -1.0, -0.4]},
    ]

    # Obstáculos cilíndricos conocidos en turtlebot3_world (x, y, radio_seguro)
    OBSTACLES = [
        (0.0, 0.0, 0.45),      # Cilindro central
        (-1.1, -1.1, 0.45),    # Cilindro SO
        (-1.1,  1.1, 0.45),    # Cilindro NO
        ( 1.1, -1.1, 0.45),    # Cilindro SE
        ( 1.1,  1.1, 0.45),    # Cilindro NE
    ]

    def __init__(self):
        super().__init__('spawn_bot')

        # Parámetros configurables
        self.declare_parameter('robot_name', 'turtlebot3_burger')
        self.declare_parameter('safety_margin', 0.45)
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.safety_margin = self.get_parameter('safety_margin').get_parameter_value().double_value

        # Clientes para /set_entity_state
        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')
        self.set_state_alt_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        # Publicador de velocidad cero para estabilizar
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publicador de la posición generada
        self.spawn_pub = self.create_publisher(Pose2D, '/spawned_pose', 10)

        # Servidor de servicio para invocar respawn desde el bucle de RL
        self.spawn_srv = self.create_service(Trigger, '/spawn_random', self.spawn_service_callback)

        self.print_free_zones()
        self.get_logger().info('Nodo spawnBot inicializado correctamente.')

    def print_free_zones(self):
        """Imprime la delimitación de zonas seguras."""
        self.get_logger().info('================ DELIMITACIÓN DE ZONAS LIBRES ================')
        self.get_logger().info(f'{"ZONA":<14} | {"X_MIN":<8} | {"X_MAX":<8} | {"Y_MIN":<8} | {"Y_MAX":<8}')
        self.get_logger().info('--------------------------------------------------------------')
        for zone in self.FREE_ZONES:
            b = zone['bounds']
            self.get_logger().info(f'{zone["name"]:<14} | {b[0]:<8.2f} | {b[1]:<8.2f} | {b[2]:<8.2f} | {b[3]:<8.2f}')
        self.get_logger().info('==============================================================')

    def is_safe_position(self, x: float, y: float) -> bool:
        """Verifica que la posición no colisione con ningún obstáculo conocido."""
        for obs_x, obs_y, safe_r in self.OBSTACLES:
            dist = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if dist < (safe_r + 0.10):
                return False
        return True

    def generate_random_safe_pose(self):
        """Genera una coordenada (x, y, yaw) aleatoria dentro de las zonas libres."""
        for _ in range(100):  # Hasta 100 intentos de muestreo seguro
            zone = random.choice(self.FREE_ZONES)
            b = zone['bounds']
            x = round(random.uniform(b[0], b[1]), 2)
            y = round(random.uniform(b[2], b[3]), 2)

            if self.is_safe_position(x, y):
                yaw = round(random.uniform(-math.pi, math.pi), 2)
                return x, y, yaw, zone['name']

        # Fallback a posición conocida segura en el sector Oeste
        return -1.80, -0.50, 0.0, 'Oeste (Fallback)'

    def spawn_robot(self, x: float = None, y: float = None, yaw: float = None):
        """
        Posiciona el robot en Gazebo en las coordenadas indicadas (o aleatorias).
        """
        if x is None or y is None:
            x, y, yaw, zone_name = self.generate_random_safe_pose()
        else:
            zone_name = 'Manual'
            if yaw is None:
                yaw = 0.0

        # Frenar robot antes de reposicionarlo
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

        # Seleccionar cliente disponible
        client = None
        if self.set_state_client.wait_for_service(timeout_sec=2.0):
            client = self.set_state_client
        elif self.set_state_alt_client.wait_for_service(timeout_sec=2.0):
            client = self.set_state_alt_client

        if client is None:
            self.get_logger().error('Servicio /set_entity_state no disponible en Gazebo.')
            return False, x, y, yaw

        # Construir solicitud de SetEntityState
        req = SetEntityState.Request()
        req.state.name = self.robot_name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.02  # Ligeramente sobre el suelo para asentar ruedas
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.linear.z = 0.0
        req.state.twist.angular.x = 0.0
        req.state.twist.angular.y = 0.0
        req.state.twist.angular.z = 0.0

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info('======================================================')
            self.get_logger().info(f'¡Robot reposicionado con éxito en Gazebo!')
            self.get_logger().info(f'• Zona elegida: {zone_name}')
            self.get_logger().info(f'• Coordenadas: x = {x:.2f} m, y = {y:.2f} m, yaw = {math.degrees(yaw):.1f}°')
            self.get_logger().info('• Estado: Sin colisión con objetos (Zona Libre validada).')
            self.get_logger().info('======================================================')

            # Publicar pose
            pose_msg = Pose2D()
            pose_msg.x = float(x)
            pose_msg.y = float(y)
            pose_msg.theta = float(yaw)
            self.spawn_pub.publish(pose_msg)
            return True, x, y, yaw
        else:
            self.get_logger().error('Fallo al reposicionar el robot con /set_entity_state.')
            return False, x, y, yaw

    def spawn_service_callback(self, request, response):
        """Callback para invocar reposicionamiento seguro por servicio."""
        success, x, y, yaw = self.spawn_robot()
        response.success = success
        response.message = f'Robot spawneado en x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SpawnBot()

    if '--service' in sys.argv or '--daemon' in sys.argv:
        node.get_logger().info('Servicio /spawn_random activo esperando solicitudes...')
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
    else:
        # Modo comando directo (one-shot): reposiciona y termina
        node.spawn_robot()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

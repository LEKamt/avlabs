#!/usr/bin/env python3
"""
Nodo: resetWorld.py
Paquete: avlab7
Descripción: Nodo para resetear el mundo en Gazebo a través de código en Python.
             Cumple con la Actividad 1.3 del Laboratorio 7.
"""

import sys
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty, Trigger
from geometry_msgs.msg import Twist


class ResetWorldNode(Node):
    """
    Nodo de ROS 2 que se conecta a los servicios de Gazebo (/reset_world y /reset_simulation)
    para reiniciar el entorno físico y los modelos durante el entrenamiento por refuerzo.
    """

    def __init__(self):
        super().__init__('reset_world_node')

        # Publicador para asegurar que el robot frene al resetear
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Clientes a servicios de Gazebo
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        self.reset_sim_client = self.create_client(Empty, '/reset_simulation')

        # Servidor de servicio personalizado para que otros nodos de RL puedan invocar el reset
        self.reset_srv = self.create_service(Trigger, '/reset_world_custom', self.reset_service_callback)

        self.get_logger().info('Nodo resetWorld inicializado.')

    def stop_robot(self):
        """Envía velocidades en cero a /cmd_vel."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def call_reset(self, simulation_time: bool = False) -> bool:
        """
        Llama al servicio de Gazebo para resetear el mundo.
        :param simulation_time: Si True, llama a /reset_simulation; si False, a /reset_world.
        """
        self.stop_robot()

        client = self.reset_sim_client if simulation_time else self.reset_world_client
        service_name = '/reset_simulation' if simulation_time else '/reset_world'

        self.get_logger().info(f'Esperando al servicio de Gazebo {service_name}...')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'Servicio {service_name} no disponible. Probando alternativa...')
            # Fallback al otro servicio
            client = self.reset_world_client if simulation_time else self.reset_sim_client
            service_name = '/reset_world' if simulation_time else '/reset_simulation'
            if not client.wait_for_service(timeout_sec=3.0):
                self.get_logger().error('No se pudo conectar a los servicios de reset de Gazebo.')
                return False

        request = Empty.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None:
            self.stop_robot()
            self.get_logger().info(f'¡Éxito! Gazebo reseteado mediante {service_name}.')
            return True
        else:
            self.get_logger().error(f'Fallo al invocar el servicio {service_name}.')
            return False

    def reset_service_callback(self, request, response):
        """Callback del servicio /reset_world_custom."""
        success = self.call_reset(simulation_time=False)
        response.success = success
        response.message = 'Mundo de Gazebo reseteado correctamente.' if success else 'Error al resetear Gazebo.'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ResetWorldNode()

    # Si se pasa --service o --daemon, se mantiene activo ofreciendo el servicio
    if '--service' in sys.argv or '--daemon' in sys.argv:
        node.get_logger().info('Servidor /reset_world_custom activo en espera de peticiones...')
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
    else:
        # Modo comando directo (one-shot): resetea y termina
        node.get_logger().info('Ejecutando reseteo del mundo Gazebo...')
        node.call_reset(simulation_time=False)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

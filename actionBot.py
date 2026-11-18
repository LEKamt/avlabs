#!/usr/bin/env python3
"""
Nodo: actionBot.py
Paquete: avlab7
Descripción: Nodo para el control y ejecución de acciones discretas del TurtleBot3
             durante el aprendizaje por refuerzo (RL) en el entorno Gazebo.
"""

import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, String, Bool


class ActionBot(Node):
    """
    Nodo de ROS 2 que suscribe a un tópico de acciones discretas y publica
    comandos de velocidad (geometry_msgs/Twist) hacia /cmd_vel para mover el TurtleBot3.
    """

    # Definición de acciones discretas (5 acciones recomendadas):
    # 0: FORWARD   (Avanzar)
    # 1: LEFT      (Girar a la izquierda)
    # 2: RIGHT     (Girar a la derecha)
    # 3: BACKWARD  (Retroceder)
    # 4: STOP      (Detenerse)
    ACTIONS = {
        0: {'name': 'FORWARD',   'linear':  0.20, 'angular':  0.00, 'desc': 'Avanzar hacia adelante'},
        1: {'name': 'LEFT',      'linear':  0.05, 'angular':  0.80, 'desc': 'Girar hacia la izquierda'},
        2: {'name': 'RIGHT',     'linear':  0.05, 'angular': -0.80, 'desc': 'Girar hacia la derecha'},
        3: {'name': 'BACKWARD',  'linear': -0.15, 'angular':  0.00, 'desc': 'Retroceder'},
        4: {'name': 'STOP',      'linear':  0.00, 'angular':  0.00, 'desc': 'Detenerse completamente'}
    }

    def __init__(self):
        super().__init__('action_bot')

        # Declaración de parámetros ROS 2
        self.declare_parameter('step_duration', 1.0)       # Duración por defecto de cada acción (segundos)
        self.declare_parameter('control_rate', 20.0)       # Frecuencia de envío de cmd_vel (Hz)
        self.declare_parameter('demo_mode', False)         # Modo demo automático

        # Detección inteligente de demo_mode tanto por parámetro ROS como por argumentos de consola (sys.argv)
        # Soporta: demo_mode:=true, demo_mode='true', demo_mode=true, --demo, demo
        argv_str = ' '.join(sys.argv).lower()
        cli_demo = ('demo' in argv_str and 'false' not in argv_str)
        param_demo = self.get_parameter('demo_mode').get_parameter_value().bool_value
        self.demo_mode = param_demo or cli_demo

        self.step_duration = self.get_parameter('step_duration').get_parameter_value().double_value
        self.control_rate = self.get_parameter('control_rate').get_parameter_value().double_value

        # Publicadores
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/action_status', 10)
        self.done_pub = self.create_publisher(Bool, '/action_done', 10)

        # Suscriptor a tópico de acciones
        self.action_sub = self.create_subscription(
            Int32,
            '/action',
            self.action_callback,
            10
        )

        # Control de estado por contador de ticks (inmune a desfases de reloj /clock / sim_time)
        self.current_action_id = 4
        self.ticks_remaining = 0
        self.target_linear = 0.0
        self.target_angular = 0.0

        # Timer de control a 20 Hz
        timer_period = 1.0 / self.control_rate
        self.control_timer = self.create_timer(timer_period, self.control_loop)

        # Imprimir información
        self.print_action_table()

        if self.demo_mode:
            self.demo_index = 0
            # Secuencia de demostración: Avanzar -> Izquierda -> Avanzar -> Derecha -> Retroceder -> Detener
            self.demo_actions = [0, 1, 0, 2, 3, 4]
            self.demo_step_seconds = 2.0
            self.get_logger().info('**************************************************************')
            self.get_logger().info('*** MODO DEMOSTRACIÓN ACTIVADO                             ***')
            self.get_logger().info('*** El robot ejecutará una secuencia cíclica de acciones. ***')
            self.get_logger().info('**************************************************************')
            # Timer de demo cada 2.5 segundos
            self.demo_timer = self.create_timer(2.5, self.demo_loop)
            # Ejecutar inmediatamente la primera acción sin esperar el timer
            self.demo_loop()
        else:
            self.get_logger().info('Nodo actionBot listo y en espera.')
            self.get_logger().info('Para mover el robot, publica una acción en /action (0 a 4).')
            self.get_logger().info('Ejemplo: ros2 topic pub --once /action std_msgs/msg/Int32 "{data: 0}"')
            self.get_logger().info('O activa el modo demo con: ros2 run avlab7 actionBot --demo')

    def print_action_table(self):
        """Imprime la tabla de acciones disponibles."""
        self.get_logger().info('================ TABLA DE ACCIONES DISCRETAS ================')
        self.get_logger().info(f'{"ID":<4} | {"ACCIÓN":<10} | {"V. Lineal (m/s)":<16} | {"V. Angular (rad/s)":<20} | {"DESCRIPCIÓN"}')
        self.get_logger().info('-------------------------------------------------------------')
        for action_id, data in self.ACTIONS.items():
            self.get_logger().info(
                f'{action_id:<4} | {data["name"]:<10} | {data["linear"]:<16.2f} | {data["angular"]:<20.2f} | {data["desc"]}'
            )
        self.get_logger().info('=============================================================')

    def execute_action(self, action_id: int, duration: float):
        """Prepara e inicia la ejecución de una acción durante una duración dada."""
        if action_id not in self.ACTIONS:
            self.get_logger().warn(f'Acción {action_id} no válida. Acciones válidas: {list(self.ACTIONS.keys())}')
            return

        action_data = self.ACTIONS[action_id]
        self.current_action_id = action_id
        self.target_linear = action_data['linear']
        self.target_angular = action_data['angular']
        self.ticks_remaining = max(1, int(duration * self.control_rate))

        self.get_logger().info(
            f'>>> Ejecutando Acción [{action_id}]: {action_data["name"]} '
            f'| v={self.target_linear:.2f} m/s, w={self.target_angular:.2f} rad/s | Duración: {duration:.1f}s'
        )

        # Publicar estado de la acción
        status_msg = String()
        status_msg.data = f'Action {action_id}: {action_data["name"]}'
        self.status_pub.publish(status_msg)

    def action_callback(self, msg: Int32):
        """Callback cuando llega un mensaje al tópico /action."""
        self.execute_action(msg.data, self.step_duration)

    def control_loop(self):
        """Bucle periódico a 20 Hz para publicar Twist en /cmd_vel."""
        if self.ticks_remaining > 0:
            twist = Twist()
            twist.linear.x = float(self.target_linear)
            twist.angular.z = float(self.target_angular)
            self.cmd_vel_pub.publish(twist)
            self.ticks_remaining -= 1

            if self.ticks_remaining == 0:
                # Al terminar los ticks, detener el robot y publicar done
                self.stop_robot()
                done_msg = Bool()
                done_msg.data = True
                self.done_pub.publish(done_msg)
                self.get_logger().info(
                    f'<<< Acción [{self.current_action_id}] completada. Robot frenado.'
                )

    def demo_loop(self):
        """Bucle de la secuencia de demostración."""
        action_to_run = self.demo_actions[self.demo_index]
        self.demo_index = (self.demo_index + 1) % len(self.demo_actions)
        self.execute_action(action_to_run, self.demo_step_seconds)

    def stop_robot(self):
        """Envía comando de frenado a /cmd_vel."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    action_bot = ActionBot()

    try:
        rclpy.spin(action_bot)
    except KeyboardInterrupt:
        action_bot.get_logger().info('Deteniendo actionBot...')
    finally:
        action_bot.stop_robot()
        action_bot.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

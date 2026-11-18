#!/usr/bin/env python3
"""
Nodo/Módulo: state_action_table.py
Paquete: avlab7
Descripción: Definición, gestión y visualización de la Tabla de Estados y Acciones (Q-Table)
             para el aprendizaje por refuerzo del TurtleBot3.
             Cumple con la Actividad 1.5 del Laboratorio 7.
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node


class StateActionTable:
    """
    Clase que encapsula la matriz Q(s, a) y los mapeos entre actuadores del robot y estados.
    """

    # Definición de las 5 acciones discretas de los actuadores (ruedas diferenciales)
    ACTIONS = {
        0: {'name': 'FORWARD',   'linear':  0.20, 'angular':  0.00, 'desc': 'Avanzar recto (+Vx)'},
        1: {'name': 'LEFT',      'linear':  0.05, 'angular':  0.80, 'desc': 'Girar a la izquierda (+Wz)'},
        2: {'name': 'RIGHT',     'linear':  0.05, 'angular': -0.80, 'desc': 'Girar a la derecha (-Wz)'},
        3: {'name': 'BACKWARD',  'linear': -0.15, 'angular':  0.00, 'desc': 'Retroceder (-Vx)'},
        4: {'name': 'STOP',      'linear':  0.00, 'angular':  0.00, 'desc': 'Frenar actuadores (0, 0)'}
    }

    def __init__(self, num_states=16, world_min=-2.0, world_max=2.0):
        self.num_actions = len(self.ACTIONS)
        self.num_states = num_states
        self.grid_size = int(np.sqrt(num_states))
        self.world_min = world_min
        self.world_max = world_max
        self.step_size = (world_max - world_min) / float(self.grid_size)

        # Inicialización de la Tabla Q con ceros (igual que en RL1.ipynb y RL2.ipynb)
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

    def coord_to_state(self, x: float, y: float) -> int:
        """Convierte coordenadas continuas (x, y) en un ID de estado discreto."""
        col = int((x - self.world_min) / self.step_size)
        row = int((y - self.world_min) / self.step_size)
        col = max(0, min(col, self.grid_size - 1))
        row = max(0, min(row, self.grid_size - 1))
        return row * self.grid_size + col

    def state_to_coord_center(self, state: int):
        """Devuelve las coordenadas del centro de la celda de un estado."""
        row = state // self.grid_size
        col = state % self.grid_size
        cx = self.world_min + (col + 0.5) * self.step_size
        cy = self.world_min + (row + 0.5) * self.step_size
        return cx, cy

    def update_q(self, state: int, action: int, reward: float, next_state: int,
                 learning_rate: float = 0.1, discount_rate: float = 0.95):
        """
        Actualización de Bellman para Q-Learning (de RL2.ipynb):
        Q[s, a] = (1 - lr) * Q[s, a] + lr * (reward + gamma * max(Q[s', :]))
        """
        best_future = np.max(self.q_table[next_state, :])
        current = self.q_table[state, action]
        self.q_table[state, action] = (1.0 - learning_rate) * current + learning_rate * (reward + discount_rate * best_future)

    def get_best_action(self, state: int) -> int:
        """Devuelve la mejor acción argmax_a Q(s, a)."""
        return int(np.argmax(self.q_table[state, :]))

    def print_table(self):
        """Imprime la tabla de estados y acciones de manera clara y tabulada."""
        print('=' * 85)
        print(f' TABLA DE ESTADOS Y ACCIONES Q(s, a) - [{self.num_states} Estados x {self.num_actions} Acciones]')
        print('=' * 85)
        header = f'{"Estado":<7} | {"Rango (x, y)":<18} | ' + ' | '.join([f'{self.ACTIONS[a]["name"][:8]:<9}' for a in range(self.num_actions)])
        print(header)
        print('-' * 85)

        for s in range(self.num_states):
            cx, cy = self.state_to_coord_center(s)
            coord_str = f'({cx:+.1f}, {cy:+.1f})'
            vals = ' | '.join([f'{self.q_table[s, a]:9.4f}' for a in range(self.num_actions)])
            print(f's_{s:<4} | {coord_str:<18} | {vals}')
        print('=' * 85)

    def save(self, filepath: str = 'q_table.npy'):
        """Guarda la tabla Q en disco."""
        np.save(filepath, self.q_table)
        print(f'Tabla Q guardada en {filepath}')

    def load(self, filepath: str = 'q_table.npy'):
        """Carga la tabla Q desde disco."""
        if os.path.exists(filepath):
            self.q_table = np.load(filepath)
            print(f'Tabla Q cargada desde {filepath}')


class StateActionTableNode(Node):
    """Nodo de ROS 2 para consultar y mostrar la tabla."""

    def __init__(self):
        super().__init__('state_action_table_node')
        self.table = StateActionTable(num_states=16)
        self.get_logger().info('Inicializando visualizador de Tabla de Estados y Acciones...')
        self.table.print_table()


def main(args=None):
    rclpy.init(args=args)
    node = StateActionTableNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

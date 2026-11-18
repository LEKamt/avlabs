#!/usr/bin/env python3
"""
Nodo: rl_agent.py
Paquete: avlab7
Descripción: Implementación de Aprendizaje por Refuerzo (PPO y Q-Learning) para el
             TurtleBot3 en Gazebo. Maneja episodios, asignación de metas distintas a
             las posiciones iniciales, función de recompensas y entrenamiento de 40+ iteraciones.
             Cumple con la Actividad 1.6 y Actividad 2.1 del Laboratorio 7.
"""

import os
import sys
import math
import time
import random
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, Pose2D
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, Bool
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty

# Intentar importar PyTorch para PPO; si no está disponible, utilizar Q-Learning / NumPy
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def euler_from_quaternion(x, y, z, w):
    """Calcula yaw a partir de cuaternión."""
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


# ==============================================================================
# REDES NEURONALES PARA PPO (ACTOR-CRITIC)
# ==============================================================================
if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """Red Actor-Crítico para PPO con espacio de acciones discretas."""
        def __init__(self, state_dim=7, action_dim=5, hidden_dim=64):
            super().__init__()
            # Actor: calcula probabilidades de cada una de las 5 acciones
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            )
            # Crítico: estima el valor del estado V(s)
            self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, state):
            probs = self.actor(state)
            value = self.critic(state)
            return probs, value

        def act(self, state):
            probs, value = self.forward(state)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action), value.squeeze(-1)

        def evaluate(self, states, actions):
            probs, values = self.forward(states)
            dist = Categorical(probs)
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            return action_log_probs, values.squeeze(-1), dist_entropy


# ==============================================================================
# NODO ENTORNO Y AGENTE RL
# ==============================================================================
class RLAgentNode(Node):
    """
    Nodo de ROS 2 que interactúa con Gazebo como un entorno RL:
    - Reinicia y posiciona el robot aleatoriamente en una zona libre.
    - Genera metas distintas al punto inicial.
    - Envía acciones a /action y observa nuevo estado y recompensa.
    - Entrena la política con PPO (o Q-Learning).
    """

    FREE_ZONES = [
        [-2.0, -1.4, -1.8,  1.8],   # Oeste
        [ 1.4,  2.0, -1.8,  1.8],   # Este
        [-1.4,  1.4,  1.4,  2.0],   # Norte
        [-1.4,  1.4, -2.0, -1.4],   # Sur
    ]

    OBSTACLES = [
        (0.0, 0.0, 0.45),
        (-1.1, -1.1, 0.45),
        (-1.1,  1.1, 0.45),
        ( 1.1, -1.1, 0.45),
        ( 1.1,  1.1, 0.45),
    ]

    def __init__(self):
        super().__init__('rl_agent_node')

        # Parámetros
        self.declare_parameter('algorithm', 'ppo')            # 'ppo' o 'qlearning'
        self.declare_parameter('num_episodes', 40)            # Mínimo 40 iteraciones (Actividad 2.1)
        self.declare_parameter('max_steps_per_episode', 40)   # Pasos máximos por episodio
        self.declare_parameter('step_time', 0.6)              # Duración de cada paso en segundos
        self.declare_parameter('robot_name', 'turtlebot3_burger')

        self.algo = self.get_parameter('algorithm').get_parameter_value().string_value.lower()
        self.num_episodes = self.get_parameter('num_episodes').get_parameter_value().integer_value
        self.max_steps = self.get_parameter('max_steps_per_episode').get_parameter_value().integer_value
        self.step_time = self.get_parameter('step_time').get_parameter_value().double_value
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value

        # Publicadores
        self.action_pub = self.create_publisher(Int32, '/action', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(Point, '/goal_pose', 10)

        # Suscriptores
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Clientes Gazebo
        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')
        self.reset_world_client = self.create_client(Empty, '/reset_world')

        # Variables internas de estado
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.has_odom = False

        self.min_laser = 3.5
        self.laser_front = 3.5
        self.laser_fl = 3.5
        self.laser_fr = 3.5
        self.laser_l = 3.5
        self.laser_r = 3.5
        self.has_scan = False

        self.goal_x = 1.5
        self.goal_y = 1.5

        # PPO Hyperparameters
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.lr = 0.002

        if TORCH_AVAILABLE and self.algo == 'ppo':
            self.policy = ActorCritic(state_dim=7, action_dim=5)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
            self.policy_old = ActorCritic(state_dim=7, action_dim=5)
            self.policy_old.load_state_dict(self.policy.state_dict())
            self.MseLoss = nn.MSELoss()
            self.get_logger().info('Inicializado Algoritmo: PPO (Actor-Critic con PyTorch).')
        else:
            self.algo = 'qlearning'
            self.q_table = np.zeros((16, 5))
            self.learning_rate = 0.1
            self.discount_rate = 0.95
            self.exploration_rate = 0.3
            self.get_logger().info('Inicializado Algoritmo: Q-Learning Tabular (basado en RL2.ipynb).')

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.has_odom = True

    def scan_callback(self, msg: LaserScan):
        ranges = msg.ranges
        n = len(ranges)
        if n == 0:
            return

        def sec_min(s_deg, e_deg):
            i1 = int((s_deg % 360) / 360.0 * n)
            i2 = int((e_deg % 360) / 360.0 * n)
            sec = ranges[i1:i2+1] if i1 <= i2 else ranges[i1:] + ranges[:i2+1]
            valid = [r for r in sec if msg.range_min < r < msg.range_max]
            return min(valid) if valid else msg.range_max

        self.laser_front = min(sec_min(345, 360), sec_min(0, 15))
        self.laser_fl = sec_min(15, 60)
        self.laser_l = sec_min(60, 120)
        self.laser_fr = sec_min(300, 345)
        self.laser_r = sec_min(240, 300)

        valid_all = [r for r in ranges if msg.range_min < r < msg.range_max]
        self.min_laser = min(valid_all) if valid_all else msg.range_max
        self.has_scan = True

    def is_safe(self, x, y):
        for ox, oy, r in self.OBSTACLES:
            if math.sqrt((x - ox)**2 + (y - oy)**2) < (r + 0.10):
                return False
        return True

    def generate_safe_point(self):
        for _ in range(50):
            z = random.choice(self.FREE_ZONES)
            x = round(random.uniform(z[0], z[1]), 2)
            y = round(random.uniform(z[2], z[3]), 2)
            if self.is_safe(x, y):
                return x, y
        return -1.8, -0.5

    def reset_episode(self):
        """
        Reinicia el episodio:
        1. Posiciona el robot en una zona libre aleatoria.
        2. Genera una meta DIFERENTE al punto de inicio (a al menos 1.8 metros).
        """
        # Frenar robot
        self.cmd_vel_pub.publish(Twist())

        # 1. Punto inicial aleatorio en zona libre
        spawn_x, spawn_y = self.generate_safe_point()
        spawn_yaw = round(random.uniform(-math.pi, math.pi), 2)

        # 2. Generar meta DIFERENTE a la posición inicial
        for _ in range(50):
            gx, gy = self.generate_safe_point()
            dist = math.sqrt((gx - spawn_x)**2 + (gy - spawn_y)**2)
            if dist >= 1.8:  # Asegurar distancia mínima significativa
                self.goal_x, self.goal_y = gx, gy
                break

        # Publicar meta
        goal_msg = Point()
        goal_msg.x = float(self.goal_x)
        goal_msg.y = float(self.goal_y)
        self.goal_pub.publish(goal_msg)

        # Reposicionar robot en Gazebo con /set_entity_state
        if self.set_state_client.wait_for_service(timeout_sec=2.0):
            req = SetEntityState.Request()
            req.state.name = self.robot_name
            req.state.pose.position.x = float(spawn_x)
            req.state.pose.position.y = float(spawn_y)
            req.state.pose.position.z = 0.02
            req.state.pose.orientation.z = math.sin(spawn_yaw / 2.0)
            req.state.pose.orientation.w = math.cos(spawn_yaw / 2.0)
            req.state.twist.linear.x = 0.0
            req.state.twist.angular.z = 0.0
            future = self.set_state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        # Esperar estabilización física
        time.sleep(0.3)
        rclpy.spin_once(self, timeout_sec=0.2)

        return self.get_state_observation()

    def get_state_observation(self):
        """Calcula el vector de estado continuo y el estado discreto."""
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        angle = math.atan2(dy, dx) - self.robot_yaw
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi

        # Vector continuo normalizado (dim=7)
        state_vec = np.array([
            min(dist / 5.0, 1.0),
            angle / math.pi,
            min(self.laser_front / 3.5, 1.0),
            min(self.laser_fl / 3.5, 1.0),
            min(self.laser_fr / 3.5, 1.0),
            min(self.laser_l / 3.5, 1.0),
            min(self.laser_r / 3.5, 1.0)
        ], dtype=np.float32)

        # Estado discreto en cuadrícula 4x4
        col = max(0, min(int((self.robot_x + 2.0) / 1.0), 3))
        row = max(0, min(int((self.robot_y + 2.0) / 1.0), 3))
        discrete_state = row * 4 + col

        return state_vec, discrete_state, dist

    def step(self, action: int, prev_dist: float):
        """
        Ejecuta la acción en el entorno, espera step_time y calcula la recompensa.
        """
        # Publicar acción al nodo actionBot
        act_msg = Int32()
        act_msg.data = action
        self.action_pub.publish(act_msg)

        # Esperar ejecución física
        time.sleep(self.step_time)
        rclpy.spin_once(self, timeout_sec=0.1)

        # Observar nuevo estado
        next_state_vec, next_discrete, new_dist = self.get_state_observation()

        # Cálculo de recompensa
        done = False
        reward = 0.0

        # 1. Recompensa por alcanzar la meta
        if new_dist < 0.35:
            reward = 100.0
            done = True
            self.get_logger().info('>>> ¡META ALCANZADA! (+100 Recompensa) <<<')
        # 2. Penalización por colisión
        elif self.min_laser < 0.22:
            reward = -100.0
            done = True
            self.get_logger().info('>>> ¡COLISIÓN DETECTADA! (-100 Penalización) <<<')
        # 3. Recompensa por progreso hacia la meta y costo por paso
        else:
            reward_progress = (prev_dist - new_dist) * 15.0
            reward_step = -0.1
            reward = reward_progress + reward_step

        return next_state_vec, next_discrete, reward, done, new_dist

    def train_ppo_update(self, states, actions, log_probs, rewards, is_terminals):
        """Actualización de la política PPO mediante Clipped Surrogate Loss."""
        if not TORCH_AVAILABLE:
            return

        # Calcular retornos descontados (Monte Carlo Returns)
        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalizar retornos
        rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(rewards_tensor) > 1 and rewards_tensor.std() > 1e-5:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)

        old_states = torch.tensor(np.array(states), dtype=torch.float32)
        old_actions = torch.tensor(np.array(actions), dtype=torch.int64)
        old_logprobs = torch.tensor(np.array(log_probs), dtype=torch.float32)

        # K épocas de optimización
        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)

            # Ventajas A = Retornos - V(s)
            advantages = rewards_tensor - state_values.detach()

            # PPO Clipped Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, rewards_tensor) - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copiar nuevos pesos a la política vieja
        self.policy_old.load_state_dict(self.policy.state_dict())

    def run_training(self):
        """Bucle principal de entrenamiento por 40+ iteraciones/episodios."""
        self.get_logger().info('Esperando datos iniciales de simulación...')
        while not (self.has_odom and self.has_scan):
            rclpy.spin_once(self, timeout_sec=0.2)
            time.sleep(0.1)

        self.get_logger().info('===============================================================')
        self.get_logger().info(f'INICIANDO APRENDIZAJE POR REFUERZO ({self.algo.upper()})')
        self.get_logger().info(f'• Iteraciones planeadas: {self.num_episodes} episodios')
        self.get_logger().info(f'• Pasos máximos por episodio: {self.max_steps}')
        self.get_logger().info('===============================================================')

        all_episode_rewards = []
        success_count = 0
        collision_count = 0

        # Buffers para PPO
        ppo_states, ppo_actions, ppo_logprobs, ppo_rewards, ppo_dones = [], [], [], [], []

        for episode in range(1, self.num_episodes + 1):
            state_vec, discrete_state, dist = self.reset_episode()
            episode_reward = 0.0
            done = False
            step_count = 0

            self.get_logger().info(
                f'\n--- Episodio {episode}/{self.num_episodes} --- | '
                f'Inicio: ({self.robot_x:.2f}, {self.robot_y:.2f}) -> Meta: ({self.goal_x:.2f}, {self.goal_y:.2f}) | '
                f'Distancia inicial: {dist:.2f} m'
            )

            for step in range(self.max_steps):
                step_count += 1

                # 1. Selección de acción
                if self.algo == 'ppo' and TORCH_AVAILABLE:
                    state_t = torch.tensor(state_vec, dtype=torch.float32)
                    with torch.no_grad():
                        action, log_prob, _ = self.policy_old.act(state_t)
                else:
                    # Q-Learning epsilon-greedy
                    if random.uniform(0, 1) < self.exploration_rate:
                        action = random.randint(0, 4)
                    else:
                        action = int(np.argmax(self.q_table[discrete_state, :]))

                # 2. Ejecutar acción en el entorno
                next_state_vec, next_discrete, reward, done, next_dist = self.step(action, dist)

                # 3. Guardar transición
                if self.algo == 'ppo' and TORCH_AVAILABLE:
                    ppo_states.append(state_vec)
                    ppo_actions.append(action)
                    ppo_logprobs.append(log_prob.item())
                    ppo_rewards.append(reward)
                    ppo_dones.append(done)
                else:
                    # Actualización Bellman Q-Learning
                    best_future = np.max(self.q_table[next_discrete, :])
                    self.q_table[discrete_state, action] = (
                        (1 - self.learning_rate) * self.q_table[discrete_state, action] +
                        self.learning_rate * (reward + self.discount_rate * best_future)
                    )

                episode_reward += reward
                state_vec = next_state_vec
                discrete_state = next_discrete
                dist = next_dist

                if done:
                    if reward > 50:
                        success_count += 1
                    elif reward < -50:
                        collision_count += 1
                    break

            all_episode_rewards.append(episode_reward)

            # Actualizar política PPO cada 2 episodios
            if self.algo == 'ppo' and TORCH_AVAILABLE and len(ppo_states) >= 40:
                self.train_ppo_update(ppo_states, ppo_actions, ppo_logprobs, ppo_rewards, ppo_dones)
                ppo_states, ppo_actions, ppo_logprobs, ppo_rewards, ppo_dones = [], [], [], [], []

            # Decaimiento de exploración si es Q-Learning
            if self.algo == 'qlearning':
                self.exploration_rate = max(0.05, self.exploration_rate * 0.95)

            self.get_logger().info(
                f'Fin Episodio {episode} | Pasos: {step_count} | '
                f'Recompensa Total: {episode_reward:.2f} | '
                f'Éxitos: {success_count}/{episode}'
            )

        # Resumen final de Actividad 2.1
        self.print_training_summary(all_episode_rewards, success_count, collision_count)

    def print_training_summary(self, rewards, successes, collisions):
        """Imprime el análisis de la política obtenida tras las 40+ iteraciones."""
        self.get_logger().info('===============================================================')
        self.get_logger().info('*** RESUMEN DEL APRENDIZAJE POR REFUERZO (ACTIVIDAD 2.1) ***')
        self.get_logger().info('===============================================================')
        self.get_logger().info(f'• Total Iteraciones Completadas: {len(rewards)}')
        self.get_logger().info(f'• Episodios con Éxito (Llegó a meta): {successes} ({successes/len(rewards)*100:.1f}%)')
        self.get_logger().info(f'• Episodios con Colisión: {collisions} ({collisions/len(rewards)*100:.1f}%)')
        self.get_logger().info(f'• Recompensa Promedio (últimos 10 episodios): {np.mean(rewards[-10:]):.2f}')
        self.get_logger().info(f'• Recompensa Máxima Obtenida: {np.max(rewards):.2f}')

        # Guardar resultados
        np.savetxt('training_rewards.csv', rewards, delimiter=',', header='reward', comments='')
        self.get_logger().info('• Curva de recompensas guardada en: training_rewards.csv')

        if self.algo == 'ppo' and TORCH_AVAILABLE:
            torch.save(self.policy.state_dict(), 'ppo_policy.pt')
            self.get_logger().info('• Modelo de política PPO guardado en: ppo_policy.pt')
        else:
            np.save('q_table_trained.npy', self.q_table)
            self.get_logger().info('• Tabla Q guardada en: q_table_trained.npy')

        self.get_logger().info('===============================================================')


def main(args=None):
    rclpy.init(args=args)
    agent = RLAgentNode()

    try:
        agent.run_training()
    except KeyboardInterrupt:
        agent.get_logger().info('Entrenamiento interrumpido por el usuario.')
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

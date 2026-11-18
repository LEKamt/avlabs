#!/usr/bin/env python3
"""
Launch file: actividad2.launch.py
Paquete: avlab7
Descripción: Archivo de lanzamiento para la Actividad 2 del Laboratorio 7.
             Ejecuta de manera coordinada todos los nodos necesarios para el
             aprendizaje por refuerzo (RL) del TurtleBot3:
             - actionBot (ejecutor de acciones discretas en /cmd_vel)
             - stateBot (cálculo y publicación de estados a partir de odom/scan/goal)
             - rl_agent (agente RL con PPO o Q-Learning por 40+ iteraciones)
             - Opcionalmente Gazebo turtlebot3_world (si launch_gazebo:=true)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declaración de argumentos
    algorithm_arg = DeclareLaunchArgument(
        'algorithm',
        default_value='ppo',
        description='Algoritmo de RL a utilizar: "ppo" (recomendado) o "qlearning"'
    )

    num_episodes_arg = DeclareLaunchArgument(
        'num_episodes',
        default_value='40',
        description='Número de episodios/iteraciones de entrenamiento (Actividad 2.1 pide al menos 40)'
    )

    max_steps_arg = DeclareLaunchArgument(
        'max_steps_per_episode',
        default_value='40',
        description='Máximo de pasos permitidos por episodio antes de reiniciar'
    )

    step_time_arg = DeclareLaunchArgument(
        'step_time',
        default_value='0.6',
        description='Duración en segundos de cada acción ejecutada en el robot'
    )

    launch_gazebo_arg = DeclareLaunchArgument(
        'launch_gazebo',
        default_value='false',
        description='Lanzar simulación Gazebo turtlebot3_world automáticamente (true/false)'
    )

    # Asegurar modelo burger por defecto para TurtleBot3
    set_tb3_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger')

    # Inclusión opcional de la simulación de Gazebo
    try:
        tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
        gazebo_launch_path = os.path.join(tb3_gazebo_share, 'launch', 'turtlebot3_world.launch.py')
        gazebo_sim = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch_path),
            condition=IfCondition(LaunchConfiguration('launch_gazebo'))
        )
    except Exception:
        gazebo_sim = None

    # Nodo 1: actionBot (Actuadores y acciones discretas)
    action_bot_node = Node(
        package='avlab7',
        executable='actionBot',
        name='action_bot',
        output='screen',
        parameters=[{
            'step_duration': LaunchConfiguration('step_time'),
            'control_rate': 20.0,
            'demo_mode': False,
        }]
    )

    # Nodo 2: stateBot (Observador y clasificador de estados)
    state_bot_node = Node(
        package='avlab7',
        executable='stateBot',
        name='state_bot',
        output='screen',
        parameters=[{
            'print_rate': 1.0,
            'grid_size': 4,
        }]
    )

    # Nodo 3: rl_agent (Entrenamiento PPO / Q-Learning)
    rl_agent_node = Node(
        package='avlab7',
        executable='rl_agent',
        name='rl_agent',
        output='screen',
        parameters=[{
            'algorithm': LaunchConfiguration('algorithm'),
            'num_episodes': LaunchConfiguration('num_episodes'),
            'max_steps_per_episode': LaunchConfiguration('max_steps_per_episode'),
            'step_time': LaunchConfiguration('step_time'),
            'robot_name': 'turtlebot3_burger',
        }]
    )

    # Armar lista de acciones del launch
    launch_actions = [
        set_tb3_model,
        algorithm_arg,
        num_episodes_arg,
        max_steps_arg,
        step_time_arg,
        launch_gazebo_arg,
        action_bot_node,
        state_bot_node,
        rl_agent_node
    ]

    if gazebo_sim is not None:
        launch_actions.insert(6, gazebo_sim)

    return LaunchDescription(launch_actions)

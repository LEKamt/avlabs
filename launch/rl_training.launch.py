#!/usr/bin/env python3
"""
Launch file: rl_training.launch.py (Alias de actividad2.launch.py)
Paquete: avlab7
"""

from actividad2.launch import generate_launch_description as _gen_desc
# Fallback directo importando el mismo generate_launch_description
import sys
import os
from ament_index_python.packages import get_package_share_directory

# Reutilizar el launch de actividad2
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    algorithm_arg = DeclareLaunchArgument('algorithm', default_value='ppo')
    num_episodes_arg = DeclareLaunchArgument('num_episodes', default_value='40')
    max_steps_arg = DeclareLaunchArgument('max_steps_per_episode', default_value='40')
    step_time_arg = DeclareLaunchArgument('step_time', default_value='0.6')
    launch_gazebo_arg = DeclareLaunchArgument('launch_gazebo', default_value='false')

    set_tb3_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger')

    try:
        tb3_gazebo_share = get_package_share_directory('turtlebot3_gazebo')
        gazebo_launch_path = os.path.join(tb3_gazebo_share, 'launch', 'turtlebot3_world.launch.py')
        gazebo_sim = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch_path),
            condition=IfCondition(LaunchConfiguration('launch_gazebo'))
        )
    except Exception:
        gazebo_sim = None

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

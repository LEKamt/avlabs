# Laboratorio 7
| **Alumno:** Luis Eduardo Kamt Chang

- Construir paquete
```
cd ~/ros2_ws && colcon build --packages-select avlab7 --symlink-install
```

- Sourcear
```
source install/setup.bash
```

- Matar procesos de gazebo
```
killall -9 gzserver gzclient 2>/dev/null; true
```

## Actividad 1 
- Abrir simulación Gazebo:
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

- Abrir nodo actionBot (Modo demostración automática):
```bash
ros2 run avlab7 actionBot --demo
```
O con parámetros ROS 2:
```bash
ros2 run avlab7 actionBot --ros-args -p demo_mode:=true
```

- Abrir nodo actionBot (Modo normal para RL):
```bash
ros2 run avlab7 actionBot
```

- Comandos para revisar cómo se mueve desde otra terminal:
```bash
# Acción 0: FORWARD (Avanzar)
ros2 topic pub --once /action std_msgs/msg/Int32 "{data: 0}"

# Acción 1: LEFT (Girar izquierda)
ros2 topic pub --once /action std_msgs/msg/Int32 "{data: 1}"

# Acción 2: RIGHT (Girar derecha)
ros2 topic pub --once /action std_msgs/msg/Int32 "{data: 2}"

# Acción 3: BACKWARD (Retroceder)
ros2 topic pub --once /action std_msgs/msg/Int32 "{data: 3}"

# Acción 4: STOP (Detener)
ros2 topic pub --once /action std_msgs/msg/Int32 "{data: 4}"
```

### 1.2. Nodo stateBot
Define los estados continuos y discretos del robot e imprime su estado periódicamente.
- Ejecutar nodo:
```bash
ros2 run avlab7 stateBot
```
- Cambiar la meta dinámica:
```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/Point "{x: 1.0, y: -1.0, z: 0.0}"
```

### 1.3. Nodo resetWorld
Resetea el mundo en Gazebo mediante código Python:
- Ejecutar reseteo directo:
```bash
ros2 run avlab7 resetWorld
```
- O dejar activo como servicio de reseteo:
```bash
ros2 run avlab7 resetWorld --service
```

### 1.4. Nodo spawnBot
Spawnea el robot en una posición aleatoria dentro de las zonas libres delimitadas, sin colisionar con ningún obstáculo:
- Ejecutar posicionamiento aleatorio seguro:
```bash
ros2 run avlab7 spawnBot
```
- O dejar activo como servicio:
```bash
ros2 run avlab7 spawnBot --service
```

### 1.5. Tabla de Estados y Acciones
Define la tabla Q(s, a) vinculada a los actuadores diferenciales:
- Visualizar tabla:
```bash
ros2 run avlab7 state_action_table
```

### 1.6 y Actividad 2: Aprendizaje por Refuerzo (PPO / Q-Learning)
Entrena la política de navegación por al menos 40 iteraciones/episodios.

#### Opción A: Mediante Launch File (Recomendado - Actividad 2)
Inicia de forma automática y sincronizada todos los nodos requeridos (`actionBot`, `stateBot`, `rl_agent`):
- Si ya tienes Gazebo corriendo en otra terminal:
```bash
ros2 launch avlab7 actividad2.launch.py algorithm:=ppo num_episodes:=40
```
- Para lanzar todo junto incluyendo la simulación en Gazebo:
```bash
ros2 launch avlab7 actividad2.launch.py launch_gazebo:=true algorithm:=ppo num_episodes:=40
```
- Para entrenar con Q-Learning (basado en RL1/RL2):
```bash
ros2 launch avlab7 actividad2.launch.py algorithm:=qlearning num_episodes:=40
```

#### Opción B: Ejecución de nodos individuales
```bash
# Terminal 1: actionBot
ros2 run avlab7 actionBot

# Terminal 2: stateBot
ros2 run avlab7 stateBot

# Terminal 3: Agente RL
ros2 run avlab7 rl_agent --ros-args -p algorithm:=ppo -p num_episodes:=40
```

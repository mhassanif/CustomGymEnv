# README: Custom Maze Game Environment

## Overview

This project demonstrates the creation of a custom Gymnasium environment for Reinforcement Learning (RL). The environment features a maze with obstacles, a start position, a goal, and a death-pit cell that ends the game with a penalty if the agent steps into it. The environment is designed to be compatible with **Stable-Baselines3**, making it suitable for applying RL algorithms to solve the maze.

![alt text](utils/screenshot.png)

## Demo Video

https://github.com/user-attachments/assets/4d3df787-8a36-4893-9cb7-477ae5d3b6d5

## Features

1. **Customizable Maze Layout**: Define the maze with starting (`S`), goal (`G`), obstacles (`#`), death-pits (`P`), and empty spaces (`.`).
2. **Interactive Rendering**: Visualize the environment using **Pygame**, including graphical representations of all elements (start, goal, obstacles, pits, and agent).
3. **Termination Conditions**:

   * Reaching the goal provides a reward and ends the episode.
   * Falling into a death-pit ends the episode with a penalty.
4. **Compatibility with RL Libraries**: Built to work with **Stable-Baselines3** for RL model training.

---

## Dependencies :
   * `gymnasium`
   * `pygame`
   * `numpy`

```bash
pip install gymnasium pygame numpy
```

---

## Usage

### Creating a Custom Maze

To create a custom maze, modify the `maze` configuration in `main.py`:

```python
maze = [
    ['S', '.', '.', '.'],
    ['.', '#', 'P', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]
```

* `S`: Starting position
* `G`: Goal position
* `.`: Empty space
* `#`: Obstacle
* `P`: Death-pit

### Setup Instructions

1. Clone the repository:

   ```bash
   git clone git@github.com:mhassanif/CustomGymEnv.git
   cd CustomGymEnv
   ```
2. Run the `main.py` script to test the environment:

   ```bash
   python main.py
   ```

   This will render the maze and simulate random actions by the agent.

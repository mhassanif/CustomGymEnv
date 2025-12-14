import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random


class MazeGameEnv(gym.Env):
    def __init__(self, size):
        super(MazeGameEnv, self).__init__()
        self.size = size  # size x size grid 
        self.num_obstacles = int(0.10 * size * size)  # 10% of cells
        self.num_pits = int(0.10 * size * size)  # 5% of cells

        self.start_pos = None
        self.goal_pos = None
        self.obstacle_positions = []
        self.pit_positions = []
        self.current_pos = None

        # 4 possible movements
        self.action_space = spaces.Discrete(4)

        # Observation space: agent position (normalized) + 4 one-step-ahead cell types
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            "visibility": spaces.MultiDiscrete([4] * 4),  # 4 cells (up, down, left, right), 4 possible states each
        })

        # Max steps before truncation
        self.max_steps = 30
        self.steps = 0

        # Initialize Pygame
        pygame.init()
        self.cell_size = 70
        self.screen = None

    def _generate_maze(self):
        # Generate valid positions for start and goal
        valid_positions = [
            (r, c) for r in range(self.size) for c in range(self.size)
        ]
    
        # Randomize the start position
        self.start_pos = random.choice(valid_positions)
        valid_positions.remove(self.start_pos)  # Remove the start position from valid positions
    
        # Randomize the goal position
        self.goal_pos = random.choice(valid_positions)
        valid_positions.remove(self.goal_pos)  # Remove the goal position from valid positions
    
        # Ensure some buffer around start and goal (optional)
        buffer_positions = [
            (self.start_pos[0] + dr, self.start_pos[1] + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= self.start_pos[0] + dr < self.size and 0 <= self.start_pos[1] + dc < self.size
        ] + [
            (self.goal_pos[0] + dr, self.goal_pos[1] + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= self.goal_pos[0] + dr < self.size and 0 <= self.goal_pos[1] + dc < self.size
        ]
        valid_positions = [pos for pos in valid_positions if pos not in buffer_positions]
    
        # Place obstacles
        self.obstacle_positions = random.sample(valid_positions, self.num_obstacles)
    
        # Remaining positions for pits
        remaining_positions = [
            pos for pos in valid_positions if pos not in self.obstacle_positions
        ]
        self.pit_positions = random.sample(remaining_positions, self.num_pits)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Reset environment with a new seed
        self._generate_maze()
        self.current_pos = self.start_pos
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # Move the agent
        new_pos = np.array(self.current_pos)
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self.current_pos = tuple(new_pos)

        # Increment step count
        self.steps += 1

        # Calculate reward and termination conditions
        if self.current_pos == self.goal_pos:
            reward = 1.0
            terminated = True
            truncated = False
            info = {"reason": "Goal reached!"}
        elif self.current_pos in self.pit_positions:
            reward = -1.0
            terminated = True
            truncated = False
            info = {"reason": "Fell into a pit!"}
        elif self.steps >= self.max_steps:
            reward = 0.0
            terminated = True
            truncated = True
            info = {"reason": "Time limit reached"}
        else:
            reward = -0.01  # Step penalty
            terminated = False
            truncated = False
            info = {"reason": "Keep exploring"}

        return self._get_obs(), reward, terminated, truncated, info

    def _is_valid_position(self, pos):
        row, col = pos
        if row < 0 or col < 0 or row >= self.size or col >= self.size:
            return False
        if (row, col) in self.obstacle_positions:
            return False
        return True

    def _get_obs(self):
        # Current position normalized
        position = np.array(self.current_pos, dtype=np.float32) / (self.size - 1)
        
        # One-step-ahead visibility
        visibility = [
            self._get_cell_type(self.current_pos[0] - 1, self.current_pos[1]),  # Up
            self._get_cell_type(self.current_pos[0] + 1, self.current_pos[1]),  # Down
            self._get_cell_type(self.current_pos[0], self.current_pos[1] - 1),  # Left
            self._get_cell_type(self.current_pos[0], self.current_pos[1] + 1),  # Right
        ]
        
        return {"position": position, "visibility": np.array(visibility, dtype=np.int32)}

    def _get_cell_type(self, row, col):
        """Returns the type of a cell: 0=empty, 1=obstacle, 2=pit, 3=goal."""
        if row < 0 or col < 0 or row >= self.size or col >= self.size:
            return 1  # Treat out-of-bounds as obstacle
        if (row, col) in self.obstacle_positions:
            return 1  # Obstacle
        if (row, col) in self.pit_positions:
            return 2  # Pit
        if (row, col) == self.goal_pos:
            return 3  # Goal
        return 0  # Empty

    def render(self):
        try:
            # Load and scale images
            obstacle_img = pygame.image.load("utils/obstacle.png")
            obstacle_img = pygame.transform.scale(obstacle_img, (self.cell_size, self.cell_size))

            start_img = pygame.image.load("utils/start.png")
            start_img = pygame.transform.scale(start_img, (self.cell_size, self.cell_size))

            goal_img = pygame.image.load("utils/goal.png")
            goal_img = pygame.transform.scale(goal_img, (self.cell_size, self.cell_size))

            pit_img = pygame.image.load("utils/fire.png")
            pit_img = pygame.transform.scale(pit_img, (self.cell_size, self.cell_size))

            agent_img = pygame.image.load("utils/spider.png")
            agent_img = pygame.transform.scale(agent_img, (self.cell_size, self.cell_size))
        except FileNotFoundError as e:
            print(f"Error loading images: {e}")
            return  # Skip rendering if images are unavailable

        # Initialize the screen if not already set
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))

        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Draw the environment
        for row in range(self.size):
            for col in range(self.size):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

                if (row, col) in self.obstacle_positions:
                    self.screen.blit(obstacle_img, (cell_left, cell_top))
                elif (row, col) in self.pit_positions:
                    self.screen.blit(pit_img, (cell_left, cell_top))
                elif (row, col) == self.start_pos:
                    self.screen.blit(start_img, (cell_left, cell_top))
                elif (row, col) == self.goal_pos:
                    self.screen.blit(goal_img, (cell_left, cell_top))

                # Draw the agent
                if np.array_equal(self.current_pos, [row, col]):
                    self.screen.blit(agent_img, (cell_left, cell_top))

        # Update the display
        pygame.display.update()

    def close(self):
        if pygame.get_init():
            pygame.quit()

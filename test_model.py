from MazeGameEnv import MazeGameEnv
from stable_baselines3 import PPO

# Initialize the environment
env = MazeGameEnv(size=8)

# Load the trained model
model = PPO.load("ppo_dynamic_maze_model", policy="MultiInputPolicy")


# Test the model and log observations
obs, _ = env.reset()
for step in range(1000):  
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step}:")
    print(f"Current Position: {obs['position']}")
    print(f"One-Step Visibility: {obs['visibility']}")
    print(f"Action Taken: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"Reward: {reward}")
    print(f"Info: {info['reason']}")
    print("------------------")
    
    if terminated or truncated:
        print(f"Episode finished: {info}")
        obs, _ = env.reset()

# Close the environment
env.close()

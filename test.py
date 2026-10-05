import time
from stable_baselines3 import PPO
from mario_env import create_optimized_mario_env

# 1. Create the environment
env = create_optimized_mario_env()

# 2. Load the saved model
# IMPORTANT: Check your './train/' folder! 
# Change 'best_model_10000' to whatever the highest number is in your folder.
MODEL_PATH = './train_phase2/best_model_5000'
model = PPO.load(MODEL_PATH)

print(f"Loaded {MODEL_PATH}! Booting up the game...")

# 3. Start the testing loop
obs, info = env.reset()

# ... (top of your test.py file stays the same) ...

while True:
    # 1. The AI decides the best move
    action, _states = model.predict(obs, deterministic=True)
    
    # 2. Extract the raw integer from the NumPy array!
    action = int(action)
    
    # 3. Pass that plain number into the game
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render the game window on your screen
    env.render()
    
    # Add a tiny delay so the game doesn't run at lightspeed
    time.sleep(0.01)
    
    # If Mario dies or finishes the level, restart!
    if terminated or truncated:
        obs, info = env.reset()
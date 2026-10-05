import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor # NEW: SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from mario_env import create_optimized_mario_env

# 1. The Callback (Remains unchanged)
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # NOTE: When using 8 clones, 1 "step" is actually 8 steps. 
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved model checkpoint at {model_path}")
        return True

# 2. The Clone Factory
# Stable-Baselines needs a function it can call repeatedly to build new environments
def make_env():
    def _init():
        return create_optimized_mario_env()
    return _init

# ==========================================
# WINDOWS MULTIPROCESSING SHIELD
# Everything below this line MUST be indented!
# ==========================================
if __name__ == '__main__':
    
   # Change it from './train/' to a new folder for this phase
    CHECKPOINT_DIR = './train_phase2/'
    LOG_DIR = './logs_phase2/'
    
    # We lower the save frequency a bit because the bot is taking 8x as many steps
    callback = TrainAndLoggingCallback(check_freq=5000, save_path=CHECKPOINT_DIR)

    # 3. Spawn the Clones!
    # Adjust this number! 8 is usually safe for modern CPUs. 
    # If your PC lags heavily, drop it to 4. If you have a beast CPU, try 16!
    num_cpu = 8 
    print(f"Spawning {num_cpu} Mario clones...")
    
    # Create the SubprocVecEnv (This runs them on separate CPU cores)
    env = SubprocVecEnv([make_env() for i in range(num_cpu)])
    env = VecMonitor(env)

    # 4. LOAD YOUR EXISTING BRAIN! (Update the number to your highest model)
    MODEL_PATH = './train_phase2/best_model_15000'
    print(f"Loading existing brain from {MODEL_PATH}...")
    # Loading the brain but injecting 5% forced exploration (Entropy)
    model = PPO.load(
        MODEL_PATH, 
        env=env, 
        tensorboard_log=LOG_DIR, 
        custom_objects={
            'ent_coef': 0.02, 
            'learning_rate': 0.0001  # <--- NEW: Micro-updates
        }
    )

    # 5. Resume Training!
    print("Unleashing the swarm! Check your TensorBoard...")
    model.learn(total_timesteps=2000000, callback=callback, reset_num_timesteps=False)
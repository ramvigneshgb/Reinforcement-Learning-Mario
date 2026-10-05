import cv2
import numpy as np
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

class MarioRLWrapper(gym.Wrapper):
    """ One master wrapper to handle Grayscale, Resize, Frame Skipping, Stacking, and Gym 0.26 Compatibility """
    def __init__(self, env, skip=4, shape=(84, 84)):
        super().__init__(env)
        self._skip = skip
        self._shape = shape
        
        # Explicitly define the exact shape Stable-Baselines3 wants: (4, 84, 84)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4, shape[0], shape[1]), dtype=np.uint8
        )
        self.frames = np.zeros((4, shape[0], shape[1]), dtype=np.uint8)

    def _preprocess(self, obs):
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # 2. Resize to 84x84
        resized = cv2.resize(gray, self._shape, interpolation=cv2.INTER_AREA)
        return resized

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        
        # Frame Skipping
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        # Process the final frame and add it to our stack
        processed_frame = self._preprocess(obs)
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = processed_frame

        # Return 5 values for modern Gym compatibility
        return self.frames.copy(), total_reward, done, False, info

    def reset(self, **kwargs):
        # Catch modern arguments like 'seed'
        obs = self.env.reset()
        processed_frame = self._preprocess(obs)
        
        # Fill the initial frame stack with the starting frame
        for i in range(4):
            self.frames[i] = processed_frame
            
        # Return observation and empty info dictionary
        return self.frames.copy(), {}

def create_optimized_mario_env():
    # 1. Initialize Base Mario
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    
    # THE SILVER BULLET: Strip away Gym's broken internal wrappers
    env = env.unwrapped 
    
    # 2. Restrict the action space
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # 3. Apply our Master Wrapper (which safely converts the 4 values to 5 for the AI)
    env = MarioRLWrapper(env)
    
    return env
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import math
import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SkipFrame(Wrapper):
    def __init__(self, env, skip=8):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        obs, total_reward, done, info = None, 0.0, False, {}
        for _ in range(self.skip):
            obs, reward, done, new_info = self.env.step(action)
            total_reward += reward
            info = new_info
            if done:
                break
        return obs, total_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    

class LinearEntropyCallback(BaseCallback):

    def __init__(
        self,
        max_timesteps: int,
        initial_coef: float = 0.2,
        final_coef: float = 0.01,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.max_timesteps = max_timesteps
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.verbose = verbose

    def _on_step(self) -> bool:
        current_timesteps = self.num_timesteps
        
        progress = min(current_timesteps / float(self.max_timesteps), 1.0)
        
        new_ent = self.initial_coef + progress * (self.final_coef - self.initial_coef)
        
        new_ent = max(self.final_coef, new_ent)
        
        self.model.ent_coef = new_ent
        

        self.logger.record("custom/entropy_coefficient", new_ent)
        self.logger.record("custom/entropy_decay_progress", progress)
        

        if self.verbose > 0 and current_timesteps % 100000 == 0:
            print(f"Timesteps: {current_timesteps:,} / {self.max_timesteps:,} | "
                  f"Entropy coef: {new_ent:.4f} | Progress: {progress:.1%}")
        
        return True



def make_env(skip=8, resize_shape=(128, 128), seed=None):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
    env = Monitor(env)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)  
    
    if seed is not None:

        try:
            env.seed(seed)
        except (AttributeError, NotImplementedError):
            print(f"Warning: Environment does not support seed()")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    env = SkipFrame(env, skip=skip) 
    env = GrayScaleObservation(env, keep_dim=True) 
    env = ResizeObservation(env, shape=resize_shape)
    
    # Set seed
    
    return env


def make_eval_callback(eval_freq=10000, n_eval_episodes=5,seed=None):
    def _init():
        env = make_env(skip=4, resize_shape=(128, 128),seed=seed)
        env = Monitor(env)
        return env
    
    eval_env = DummyVecEnv([_init])
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order='last')
    
        
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/',
        log_path='./callback_logs/',
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    return eval_callback

def quick_seed_test():
    """Quick test to verify if seed is working properly"""
    print("=" * 50)
    print("Testing seed effectiveness...")
    
    # Create two environments with same seed
    env1 = make_env(skip=4, resize_shape=(128, 128), seed=42)
    env2 = make_env(skip=4, resize_shape=(128, 128), seed=42)
    
    # Reset and compare initial states
    obs1 = env1.reset()
    obs2 = env2.reset()
    
    initial_same = np.array_equal(obs1, obs2)
    print(f"Initial states identical: {initial_same}")
    
    # Execute same actions
    rewards_same = True
    for i in range(5):
        obs1, r1, d1, i1 = env1.step(1)  # Same action
        obs2, r2, d2, i2 = env2.step(1)
        if r1 != r2:
            rewards_same = False
            print(f"  Step {i}: Different rewards ({r1} vs {r2})")
    
    print(f"First 5 steps rewards identical: {rewards_same}")
    
    # Create environment with different seed for comparison
    env3 = make_env(skip=4, resize_shape=(128, 128), seed=99)
    obs3 = env3.reset()
    
    different_seed = not np.array_equal(obs1, obs3)
    print(f"Different seeds produce different states: {different_seed}")
    
    env1.close()
    env2.close()
    env3.close()
    
    # Summary
    if initial_same and rewards_same:
        print("✅ Seed is likely working correctly!")
    else:
        print("⚠️ Seed may not be fully effective")
    print("=" * 50)
    
    # Optional: Ask whether to continue training
    response = input("Continue training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled")
        exit()

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from stable_baselines3 import PPO
from gym.wrappers import GrayScaleObservation
import matplotlib.pyplot as plt

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)

obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())

plt.imshow(obs.squeeze(), cmap="gray")  # Add squeeze into (H, W)
plt.title("Mario observation (grayscale)")
plt.axis("off")
plt.show()  

env.close()

from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from utils import make_env
from stable_baselines3 import PPO
from gym.wrappers import GrayScaleObservation

env = DummyVecEnv([lambda: make_env(skip=4, resize_shape=(128, 128))])
env = VecFrameStack(env, n_stack=4, channels_order='last')

model = PPO.load(path='./best_model/best_model.zip',env=env)

obs = env.reset()
for step in range(10000):
    obs = obs.copy()
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
env.close() 
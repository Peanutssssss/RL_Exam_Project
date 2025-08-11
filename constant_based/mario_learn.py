from stable_baselines3 import PPO       
from utils import make_env, make_eval_callback,  set_seed, quick_seed_test
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
import os

def main():
    
    os.makedirs("./best_model", exist_ok=True)
    os.makedirs("./callback_logs", exist_ok=True)
    
    #You are free to set the value of seed.
    SEED = 123
    set_seed(SEED)
    
    vec_env = SubprocVecEnv([
        lambda i=i: make_env(skip=4, resize_shape=(128, 128), seed=SEED+i) 
        for i in range(8)
        ])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order='last')
    
    callback = make_eval_callback(seed=SEED+1000)
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="logs",
        learning_rate=3e-4, 
        n_steps=512,
        batch_size=512,
        n_epochs=6,
        ent_coef=0.1,
        gamma=0.96,
        target_kl=0.10,
        clip_range=0.2,
        device= 'cuda',
        seed=SEED,
    )
    
    model.learn(total_timesteps=2e7, callback=callback)

if __name__ == "__main__":
    main()

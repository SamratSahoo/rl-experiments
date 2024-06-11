import time
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("FrozenLake-v1", n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=250000)
model.save("ppo_frozenlake")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_frozenlake")

obs = vec_env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    time.sleep(1)

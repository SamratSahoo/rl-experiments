from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1").learn(10000)

print(model.get_parameters())

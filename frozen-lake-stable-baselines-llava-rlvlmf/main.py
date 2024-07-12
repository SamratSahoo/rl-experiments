import base64
import io
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from PIL import Image
from llava import generate_analysis, generate_label
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker


# class FrozenLakeByteEnvironment(ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#     def observation(self, obs):
#         rgb_array = env.render()
#         image = Image.fromarray(rgb_array.astype("uint8"), "RGB")
#         byte_io = io.BytesIO()
#         image.save(byte_io, "PNG")
#         byte_data = byte_io.getvalue()

#         return byte_data


class FrozenLakeExtendedEnv(FrozenLakeEnv):
    def __init__(self, objective):
        super().__init__(render_mode="rgb_array", is_slippery=False)
        self.objective = objective

    def step(self, a):
        current_state_bytes = self.__get_state_bytes()
        next_state, reward, terminated, truncated, info = super().step(a)
        next_state_bytes = self.__get_state_bytes()

        analysis = generate_analysis(
            current_state_bytes, next_state_bytes, self.objective
        )
        reward = int(generate_label(analysis, self.objective))
        print(reward)

        return next_state, reward, terminated, truncated, info

    def __get_state_bytes(self):
        rgb_array = env.render()
        image = Image.fromarray(rgb_array.astype("uint8"), "RGB")
        byte_io = io.BytesIO()
        image.save(byte_io, "PNG")
        byte_data = byte_io.getvalue()

        return byte_data


if __name__ == "__main__":

    train_mode = True

    if train_mode:
        env = FrozenLakeExtendedEnv(
            objective="get the elf to the giftbox while avoiding the ponds"
        )
        current_state, _ = env.reset()

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10)

        model.save("model.zip")
        env.close()
    else:
        env = gym.make(
            "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human"
        )

        model = PPO("MlpPolicy", env, verbose=1)
        model.load("model.zip")
        current_state, _ = env.reset()
        for step in range(500):
            env.render()
            next_state, reward, done, truncated, info = env.step(
                model.predict(current_state)
            )
            current_state = next_state

            if done or truncated:
                break

        env.close()

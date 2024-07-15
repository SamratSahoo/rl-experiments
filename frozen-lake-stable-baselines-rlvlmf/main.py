import io
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from PIL import Image
from gpt import generate_analysis, generate_label
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import glob
from pathlib import Path


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
        return next_state, reward, terminated, truncated, info

    def __get_state_bytes(self):
        rgb_array = env.render()
        image = Image.fromarray(rgb_array.astype("uint8"), "RGB")
        byte_io = io.BytesIO()
        image.save(byte_io, "PNG")
        byte_data = byte_io.getvalue()

        return byte_data


def get_most_recent_file(folder_path):
    files = glob.glob(os.path.join(folder_path, "*"))
    if not files:
        return None

    most_recent_file = max(files, key=os.path.getctime)

    return os.path.basename(most_recent_file)


if __name__ == "__main__":
    train_mode = False

    if train_mode:
        env = FrozenLakeExtendedEnv(objective="move the elf to the giftbox")
        current_state, _ = env.reset()

        checkpoint_callback = CheckpointCallback(
            save_freq=50,
            save_path="./models/",
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        if get_most_recent_file("./models"):
            model = PPO.load(
                Path(f"./models/{get_most_recent_file('./models/')}").with_suffix(""),
                env=env,
                device="cuda",
            )
        else:
            model = PPO("MlpPolicy", env, verbose=1)

        model.learn(
            total_timesteps=1000,
            progress_bar=True,
            callback=checkpoint_callback,
        )

        model.save("model.zip")
        env.close()
    else:

        env = gym.make(
            "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human"
        )

        model = PPO("MlpPolicy", env, verbose=1)
        model.load(
            Path(f"./models/{get_most_recent_file('./models/')}").with_suffix("")
        )
        current_state, _ = env.reset()
        for step in range(500):
            env.render()
            next_state, reward, done, truncated, info = env.step(
                int(model.predict(current_state)[0])
            )
            current_state = next_state

            if done or truncated:
                break

        env.close()

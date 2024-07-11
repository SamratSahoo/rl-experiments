import base64
import io
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from PIL import Image
from llava import generate_analysis, generate_label

env = gym.make(
    "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array"
)


class FrozenLakeB64Environment(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        rgb_array = env.render()
        image = Image.fromarray(rgb_array.astype("uint8"), "RGB")
        byte_io = io.BytesIO()
        image.save(byte_io, "PNG")
        byte_data = byte_io.getvalue()

        return byte_data


if __name__ == "__main__":
    env = FrozenLakeB64Environment(env)
    current_state, _ = env.reset()
    action_map = {0: "left", 1: "down", 2: "right", 3: "up", -1: "n/a"}
    action = env.action_space.sample()
    print(f"Action Taken: {action_map[action]}")

    next_state, reward, done, truncated, _ = env.step(action)

    analysis = generate_analysis(
        current_state,
        next_state,
        objective="get the elf to the giftbox while avoiding the ponds",
    )

    print(analysis)

    label = generate_label(
        analysis,
        objective="get the elf to the giftbox while avoiding the ponds",
    )

    print(label)

import gymnasium as gym
import random
from operator import itemgetter


num_action_map = {0: "left", 1: "down", 2: "right", 3: "up", -1: "n/a"}


def train(
    policy,
    slippery=False,
    map_name="4x4",
    steps=1000,
    learning_rate=0.9,
    discount_factor=0.9,
    epsilon=1,
    decay=0.0001,
    show_debug=True,
    render=False,
):
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=slippery,
        render_mode=("human" if render else None),
    )
    current_state, _ = env.reset()

    def get_action(epsilon, current_state):
        # Choose random action if we haven't set an action yet
        if policy[current_state] == -1:
            action = env.action_space.sample()
        else:
            # Set policy to be epsilon greedy
            if random.uniform(0, 1) > 1 - epsilon:
                action = env.action_space.sample()
            else:
                action = policy[current_state]

        return action

    next_state = 0

    q = [
        [0 for _ in range(env.action_space.n)]
        for _ in range(env.observation_space.n * env.observation_space.n)
    ]

    for t in range(1, steps + 1):
        epsilon = max(0, epsilon - decay)
        action = get_action(epsilon, current_state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        q[current_state][action] = q[current_state][action] + learning_rate * (
            reward
            + discount_factor
            * max(
                [
                    q[next_state][possible_action]
                    for possible_action in range(env.action_space.n)
                ]
            )
            - q[current_state][action]
        )

        if terminated or truncated:
            current_state, _ = env.reset()
        else:
            current_state = next_state

        new_action, _ = max(enumerate(q[current_state]), key=itemgetter(1))
        policy[current_state] = new_action

        if t % 1000 == 0 and show_debug:
            print("Current Policy:", [num_action_map[p] for p in policy])


def simulate(policy, slippery=False, map_name="4x4"):
    terminated = False
    truncated = False
    env = gym.make(
        "FrozenLake-v1", render_mode="human", map_name=map_name, is_slippery=slippery
    )
    current_state, _ = env.reset()

    while not terminated and not truncated:
        action = policy[current_state]
        next_state, _, terminated, truncated, _ = env.step(action)
        current_state = next_state

    if truncated:
        print("Simulation Failed")
    else:
        if current_state == 15:
            print("Simulation Passed")
        else:
            print("Simulation Failed")

    env.close()


def evaluate(policy, slippery=False, map_name="4x4"):
    passed = 0
    failed = 0

    for i in range(0, 100):
        terminated = False
        truncated = False
        env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery)
        current_state, _ = env.reset()

        while not terminated and not truncated:
            action = policy[current_state]
            next_state, _, terminated, truncated, _ = env.step(action)
            current_state = next_state

        if truncated:
            failed += 1
        else:
            if current_state == 15:
                passed += 1
            else:
                failed += 1

    env.close()
    print(f"Success Rate = {100*passed/(passed + failed)}%")


if __name__ == "__main__":
    slippery = True
    map_name = "4x4"
    steps = 1500000
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1
    decay = 0.0001

    temp_env = gym.make("FrozenLake-v1", map_name=map_name)
    policy = [-1 for _ in range(temp_env.observation_space.n)]

    train(
        policy,
        slippery=slippery,
        map_name=map_name,
        steps=steps,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        decay=decay,
        show_debug=False,
        render=False,
    )
    evaluate(policy, slippery=slippery, map_name=map_name)
    simulate(policy, slippery=slippery, map_name=map_name)

import gymnasium as gym
import random
from operator import itemgetter

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
observation, info = env.reset()


q = [[0 for _ in range(4)] for _ in range(16)]
policy = [-1 for _ in range(4*4)]
num_action_map = {0: "left", 1: "down", 2: "right", 3: "up", -1: "n/a" }

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

def get_adjusted_reward(current_state, next_state, action):
    bad_states = [5, 7, 11, 12]
    left_border_states = [0, 4, 8, 12]
    right_border_states = [3, 7, 11, 15]
    top_border_states = [0,1,2,3]
    bottom_border_states = [12,13,14,15]

    if next_state == 15:
        return 5

    if next_state in bad_states:
        return -5
    
    if num_action_map[action] == "down" and current_state in bottom_border_states:
        return -2
    if num_action_map[action] == "up" and current_state in top_border_states:
        return -2
    if num_action_map[action] == "left" and current_state in left_border_states:
        return -2
    if num_action_map[action] == "right" and current_state in right_border_states:
        return -2

    return 0


def train():

    learning_rate = 0.01
    discount_factor = 1
    current_state = 0
    next_state = 0
    steps = 800
    epsilon = 1
    decay = 0.001

    for t in range(1, steps + 1):
        epsilon = max(0, epsilon - decay)
        action = get_action(epsilon, current_state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        reward = get_adjusted_reward(current_state, next_state, action)

        q[current_state][action] = q[current_state][action] + learning_rate * (
            reward
            + discount_factor
            * max(
                [q[next_state][possible_action] for possible_action in range(4)]
            )
            - q[current_state][action]
        )


        if terminated or truncated:
            observation, info = env.reset()
            current_state = 0
        else:
            current_state = next_state
            new_action, _ = max(enumerate(q[current_state]), key=itemgetter(1))
            policy[current_state] = new_action

        if t % 100 == 0:
            print("Current Policy:", [num_action_map[p] for p in policy])
        
def simulate():
    terminated = False
    truncated = False
    env2 = gym.make("FrozenLake-v1", render_mode="human", map_name="4x4", is_slippery=False)
    current_state, _ = env2.reset()

    while not terminated and not truncated:
        action = policy[current_state]
        next_state, _, terminated, truncated, _ = env2.step(action)
        current_state = next_state

    if truncated:
        print("Simulation Failed")
    else:
        if current_state == 15:
            print("Simulation Passed")
        else:
            print("Simulation Failed")

    env.close()

def evaluate():
    passed = 0
    failed = 0

    for i in range(0, 10000):
        terminated = False
        truncated = False
        env2 = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
        current_state, _ = env2.reset()

        while not terminated and not truncated:
            action = policy[current_state]
            next_state, _, terminated, truncated, _ = env2.step(action)
            current_state = next_state

        if truncated:
            failed += 1
        else:
            if current_state == 15:
                passed += 1
            else:
                failed += 1

        env.close()
    print (f"Success Rate = {100*passed/(passed + failed)}%")


if __name__ == "__main__":
    train()
    evaluate()
    simulate()

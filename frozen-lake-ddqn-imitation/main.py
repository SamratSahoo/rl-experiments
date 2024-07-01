# Largely Based on: https://github.com/MehdiShahbazi/DQN-Frozenlake-Gymnasium/blob/master/DQN.py

from collections import defaultdict, deque
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class ExtendedFrozenLake(FrozenLakeEnv):
    def __init__(self, render_mode="rgb_array"):
        FrozenLakeEnv.__init__(self, render_mode=render_mode)
        self.s = 15
        self.step(1)
        self.reset()

    def reset(self):
        self.s = 15


class ReplayBuffer:
    def __init__(self, max_size=15000):
        self.max_size = max_size
        self.states = deque([], maxlen=max_size)
        self.actions = deque([], maxlen=max_size)
        self.rewards = deque([], maxlen=max_size)
        self.next_states = deque([], maxlen=max_size)
        self.dones = deque([], maxlen=max_size)

    def append(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, k):
        indices = np.random.choice(len(self), size=k, replace=False)

        sampled_states = torch.stack(
            [
                torch.as_tensor(self.states[i], dtype=torch.float32, device=device)
                for i in indices
            ]
        ).to(device)
        sampled_actions = torch.as_tensor(
            [self.actions[i] for i in indices], dtype=torch.long, device=device
        )
        sampled_rewards = torch.as_tensor(
            [self.rewards[i] for i in indices], dtype=torch.float32, device=device
        )
        sampled_next_states = torch.stack(
            [
                torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device)
                for i in indices
            ]
        ).to(device)
        sampled_dones = torch.as_tensor(
            [self.dones[i] for i in indices], dtype=torch.bool, device=device
        )

        return (
            sampled_states,
            sampled_actions,
            sampled_rewards,
            sampled_next_states,
            sampled_dones,
        )

    def __len__(self):
        return len(self.actions)


class DQN(nn.Module):
    def __init__(self, input_dims, output_dims, lr=1e-4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dims)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Agent:
    def __init__(self, hyperparameters=defaultdict(int)):
        self.losses = []
        self.rewards = []

        self.epsilon = hyperparameters["epsilon"] or 1.0
        self.epsilon_floor = hyperparameters["epsilon_floor"] or 0.01
        self.replay_buffer_capacity = hyperparameters["replay_buffer_capacity"] or 10000

        self.batch_size = hyperparameters["batch_size"] or 32
        self.episodes = hyperparameters["episodes"] or 3000
        self.epsilon_decay = hyperparameters["epsilon_decay"] or 1 / (self.episodes)

        self.environment = hyperparameters["environment"] or gym.make(
            "FrozenLake-v1", map_name="4x4", is_slippery=False
        )
        self.render_environment = hyperparameters["render_environment"] or gym.make(
            "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array"
        )
        self.simulation_environment = hyperparameters[
            "simulation_environment"
        ] or gym.make(
            "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human"
        )

        self.goal_environment = (
            hyperparameters["goal_environment"] or ExtendedFrozenLake()
        )

        self.learning_rate = hyperparameters["learning_rate"] or 1e-2
        self.update_frequency = hyperparameters["update_frequency"] or 100
        self.save_frequnecy = hyperparameters["save_frequency"] or 1000
        self.discount_factor = hyperparameters["discount_factor"] or 0.95
        self.model_save_path = hyperparameters["model_save_path"] or "./model.pth"
        self.model_load_path = hyperparameters["model_load_path"] or "./model.pth"
        self.log_frequency = hyperparameters["model_save_path"] or 1000
        self.observation_space_size = self.environment.observation_space.n
        self.action_space = self.environment.action_space
        self.action_space_size = self.environment.action_space.n
        self.replay_buffer = ReplayBuffer(max_size=self.replay_buffer_capacity)

        self.value_dqn = DQN(
            torch.numel(self.encode_state(self.action_space.sample())),
            self.action_space_size,
        ).to(device)
        self.target_dqn = DQN(
            torch.numel(self.encode_state(self.action_space.sample())),
            self.action_space_size,
        ).to(device)

        self.target_dqn.load_state_dict(self.value_dqn.state_dict())

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.value_dqn.parameters(), lr=self.learning_rate)

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        dones = dones.unsqueeze(1)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)

        pred_q = self.value_dqn(states).gather(dim=1, index=actions)
        with torch.no_grad():
            next_target_q_value = self.target_dqn(next_states).max(dim=1, keepdim=True)[
                0
            ]
        next_target_q_value[dones] = 0

        target_q = rewards + self.discount_factor * next_target_q_value
        loss = self.loss_fn(pred_q, target_q)
        self.losses.append(float(loss))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state, inference=False):
        if inference:
            return torch.argmax(self.target_dqn(state)).item()

        if np.random.random() < self.epsilon:
            return self.action_space.sample()

        with torch.no_grad():
            return torch.argmax(self.value_dqn(state)).item()

    def train(self):
        goal_state = self.get_goal_state()
        for episode in range(1, self.episodes + 1):
            current_state, _ = self.environment.reset()
            current_state_t, _ = self.environment.reset()
            current_state = self.encode_state(current_state)
            done = False
            truncated = False
            current_episode_reward = 0
            current_episode_steps = 0

            while not done and not truncated:
                action = self.choose_action(current_state)
                next_state, reward, done, truncated, _ = self.environment.step(action)
                next_state_t, _, _, _, _ = self.render_environment.step(action)
                reward = self.similarity(next_state, goal_state)
                next_state = self.encode_state(next_state)

                self.replay_buffer.append(
                    current_state, action, reward, next_state, done
                )

                if len(self.replay_buffer) > self.batch_size and sum(self.rewards):
                    self.learn()

                    if episode % self.update_frequency == 0:
                        self.update_target_model()

                current_episode_reward += reward
                current_state = next_state
                current_episode_steps += 1
            self.rewards.append(current_episode_reward)
            self.update_epsilon()

            if episode % self.save_frequnecy:
                torch.save(self.target_dqn.state_dict(), self.model_save_path)

            if episode % self.log_frequency == 0:
                self.plot_losses()
                self.plot_rewards()

            print(
                f"Episode: {episode}, Current Episode Steps: {current_episode_steps}, Reward: {current_episode_reward}, Epsilon: {self.epsilon}"
            )

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_floor, self.epsilon - self.epsilon_decay)

    def update_target_model(self):
        self.target_dqn.load_state_dict(self.value_dqn.state_dict())

    def encode_state(self, state):
        one_hot = torch.zeros(
            self.observation_space_size, dtype=torch.float32, device=device
        )
        one_hot[state] = 1

        return one_hot

    def get_goal_state(self):
        self.goal_environment.reset()
        img = self.goal_environment.render()
        return np.array(img)

    def plot_losses(self):
        plt.figure(1)
        plt.plot(self.losses)
        plt.savefig("loss.png")
        plt.clf()

    def plot_rewards(self):
        plt.figure(2)
        plt.plot(self.rewards)
        plt.savefig("reward.png")
        plt.clf()

    def similarity(self, A, B):
        mse = max(np.sum(np.square(A - B)), 0.01)
        return 1 / mse

    def simulate(self):
        done = False
        truncated = False
        current_state, _ = self.simulation_environment.reset()
        current_state = self.encode_state(current_state)
        self.target_dqn.load_state_dict(torch.load(self.model_load_path))
        self.target_dqn.eval()

        while not done and not truncated:
            action = self.choose_action(current_state, inference=True)
            next_state, _, done, truncated, _ = self.simulation_environment.step(action)
            self.simulation_environment.render()
            next_state = self.encode_state(next_state)
            current_state = next_state


if __name__ == "__main__":
    agent = Agent()

    agent.train()
    # agent.simulate()

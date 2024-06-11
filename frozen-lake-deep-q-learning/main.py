import gymnasium as gym
import os

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from itertools import count

from collections import deque

env = gym.make(
    "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class DQN(nn.Module):
    def __init__(self, input_dims, output_dim, lr):
        super(DQN, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 1, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1)

        self.fc1 = nn.Linear(32, 512)
        self.fc2 = nn.Linear(512, self.output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Agent:
    def __init__(
        self,
        input_dim,
        output_dim,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        eps_dec=1e-5,
        eps_min=0.01,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.output_dim)]

        self.Q = DQN(self.input_dim, self.output_dim, self.lr)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation, dtype=torch.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = torch.argmax(actions).item() // 64
        else:
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        self.epsilon -= self.eps_dec
        self.epsilon = max(self.epsilon, self.eps_min)

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = torch.tensor(state, dtype=torch.float).to(self.Q.device)
        actions = torch.tensor(action).to(self.Q.device)
        rewards = torch.tensor(reward).to(self.Q.device)
        states_ = torch.tensor(state_, dtype=torch.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]

        q_next = self.Q.forward(states_).max()

        q_target = rewards + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()


def train():
    env.reset()
    current_state_t = torch.as_tensor(env.render())
    input_size = current_state_t.shape
    output_size = env.action_space.n
    agent = Agent(input_dim=input_size, output_dim=output_size)
    iterations = 10001

    scores = []
    avg_scores = []
    eps_history = []
    for i in range(iterations):
        score = 0
        done = False

        env.reset()
        obs = torch.as_tensor(env.render())
        step_count = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, trunc, term, info = env.step(action)
            done = term or trunc

            if reward > 0:
                reward += 10000
            elif not done:
                reward += obs_ - step_count
            else:
                reward -= 100

            obs_ = torch.as_tensor(env.render())
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
            step_count += 1

        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            print(
                "episode ",
                i,
                "score %.1f avg score %.5f epsilon %.2f"
                % (score, avg_score, agent.epsilon),
            )

            plt.plot([x * 100 for x in range(len(avg_scores))], avg_scores)
            plt.savefig("scores.png")

            model_path = "model.pth"
            torch.save(agent.Q.state_dict(), model_path)


def simulate():
    env.reset()
    current_state_t = torch.as_tensor(env.render())
    input_size = current_state_t.shape
    output_size = env.action_space.n
    agent = Agent(input_dim=input_size, output_dim=output_size, epsilon=0)

    final_model_path = "model.pth"
    agent.Q.load_state_dict(torch.load(final_model_path))
    agent.Q.eval()

    terminated = False
    truncated = False
    env2 = gym.make(
        "FrozenLake-v1", render_mode="human", map_name="4x4", is_slippery=False
    )
    env2.reset()

    while not terminated and not truncated:
        action = agent.choose_action(current_state_t)
        next_state, _, terminated, truncated, _ = env.step(action)
        env2.step(action)
        env2.render()
        current_state_t = torch.as_tensor(env.render())

    env.close()
    env2.close()


if __name__ == "__main__":
    train()
    simulate()

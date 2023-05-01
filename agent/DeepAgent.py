import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from random import randint, random


def transform_state(observation, n_item):
    n, m = observation.shape
    state = torch.zeros(n_item + 1, n, m)
    for i in range(n):
        for j in range(m):
            item_idx = observation[i][j]
            state[item_idx][i][j] = 1
    return state


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = nn.functional.relu(self.fc1(state))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepAgent:
    def __init__(self, state_size, action_size, hidden_size, learning_rate, gamma, epsilon, epsilon_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)
        self.qnetwork = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)

    def act(self, state):
        if random() < self.epsilon:
            return randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.qnetwork(state)
                return np.argmax(q_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = list(self.memory)[-batch_size:]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.qnetwork(states)
        next_q_values = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values.gather(1, actions.unsqueeze(1)), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon *= self.epsilon_decay

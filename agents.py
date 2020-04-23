import math
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from model import QNetwork, DuelingNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

BUFFER_SIZE = 100000
BATCH_SIZE = 64
UPDATE_TARGET_STEPS = 1000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
LR = 0.0001

Transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state", "done"])

class DQN_Agent:
    def __init__(self, state_size, action_size, seed=42):
        self.action_size = action_size

        # Q-Network
        self.q_eval = QNetwork(state_size, action_size, seed).to(device)
        self.q_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.RMSprop(self.q_eval.parameters(), lr=LR)

        # Replay Buffer
        self.memory = ReplayBuffer(seed=seed)

        self.step_count = 0
        self.seed = random.seed(seed)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_eval.eval()
        with torch.no_grad():
            q_values = self.q_eval(state)
        self.q_eval.train()

        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.step_count / EPS_DECAY)
        if random.random() > epsilon:
            # greedy
            return np.argmax(q_values.cpu().data.numpy())
        else:
            # explore
            return random.choice(np.arange(self.action_size))
        
    def step(self, state, action, reward, next_state, done):       
        self.memory.push(state, action, reward, next_state, done)

        loss_value = None
        if len(self.memory) >= BATCH_SIZE:
            # sample transitions from replay buffer
            states, actions, rewards, next_states, dones = self.memory.sample()
 
            #  r                                   if done
            #  r + max_a \gamma Q(s, a; \theta')   if not done
            q_next_values = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
            q_learning_targets = rewards + GAMMA * q_next_values * (1 - dones)

            # Q(s, a; \theta)
            q_values = self.q_eval(states).gather(1, actions)

            # perform gradient descent on the loss
            loss = F.mse_loss(q_values, q_learning_targets)
            loss_value = loss.data.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update target Q-Network
            self.update_target()

        self.step_count += 1
        return loss_value
    
    def update_target(self):
        if self.step_count % UPDATE_TARGET_STEPS == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

class DoubleDQN_Agent(DQN_Agent):
    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

        loss_value = None
        if len(self.memory) >= BATCH_SIZE:
            # sample transitions from replay buffer
            states, actions, rewards, next_states, dones = self.memory.sample()
 
            # Double DQN
            #  r                                                          if done
            #  r + \gamma * Q(s', argmax_a' Q(s', a'; \theta); \theta' )  if not done
            with torch.no_grad():
                q_values = self.q_eval(next_states)
            next_state_actions = q_values.argmax(1).unsqueeze(1)
            q_next_values = self.q_target(next_states).gather(1, next_state_actions)

            q_learning_targets = rewards + GAMMA * q_next_values * (1 - dones)

            # Q(s, a; \theta)
            q_values = self.q_eval(states).gather(1, actions)

            # perform gradient descent on the loss
            loss = F.mse_loss(q_values, q_learning_targets)
            loss_value = loss.data.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update target Q-Network
            self.update_target()

        self.step_count += 1
        return loss_value

class DuelingNetwork_Agent(DQN_Agent):
    def __init__(self, state_size, action_size, seed=42):
        self.action_size = action_size

        # Q-Network
        self.q_eval = DuelingNetwork(state_size, action_size, seed).to(device)
        self.q_target = DuelingNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.RMSprop(self.q_eval.parameters(), lr=LR)

        # Replay Buffer
        self.memory = ReplayBuffer(seed=seed)

        self.step_count = 0
        self.seed = random.seed(seed)

class ReplayBuffer:
    def __init__(self, seed):
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.seed = random.seed(seed)
    
    def push(self, *args):
        # append a Transition
        self.memory.append(Transition(*args))
    
    def sample(self):
        # sample and return a batch of transitions
        transitions = random.sample(self.memory, k=BATCH_SIZE)

        states = torch.from_numpy(np.vstack([t.state for t in transitions])).float().to(device)
        actions = torch.from_numpy(np.vstack([t.action for t in transitions])).long().to(device)
        rewards = torch.torch.from_numpy(np.vstack([t.reward for t in transitions])).float().to(device)
        next_states = torch.from_numpy(np.vstack([t.next_state for t in transitions])).float().to(device)
        dones = torch.from_numpy(np.vstack([t.done for t in transitions]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
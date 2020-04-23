import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=256):
        super(DuelingNetwork, self).__init__()
        self.action_size = action_size

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)

        self.fc2_val = nn.Linear(fc1_units, fc2_units)
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)

        self.fc3_val = nn.Linear(fc2_units, 1)
        self.fc3_adv = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))

        val = F.relu(self.fc2_val(x))
        adv = F.relu(self.fc2_adv(x))

        val = self.fc3_val(val)
        adv = self.fc3_adv(adv)

        adv_avg = torch.mean(adv, dim=1, keepdim=True)

        return val + adv - adv_avg

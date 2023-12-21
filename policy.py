import torch
from torch_util import Module
import torch.nn as nn
import torch.nn.functional as F


class Actor(Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=256, min_log_std=-10.0, max_log_std=10.0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)

        self.mu_head = nn.Linear(hidden_sizes, action_dim)
        self.log_std_head = nn.Linear(hidden_sizes, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(torch.clamp(self.log_std_head(x), self.min_log_std, self.max_log_std)).sqrt()

        action = mu + std * torch.randn_like(mu)
        action = torch.clamp(action, -1.0, 1.0)

        return mu, std, action







import torch
import torch.nn as nn
import torch.nn.functional as F

"""
- QNet defines a NN w/ a simple two hidden layer structure, but for complex problems we need a better structure
- In DQL, its useful to have two separate networks, one for the current Q-values (local network) and one for the target Q-values (target network). Both these networks have the same archiecture, but different params.
- # of nodes in each hidden layer is parameterized with `fc1` and `fc2`, giving more flexibility to experiement w/ different architectures w/ out modifying the class structure
"""

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

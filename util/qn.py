"""Dueling Q-network used by the Banana navigation agent.

The network maps a state vector to one action value per discrete action.
It uses the dueling architecture (Wang et al., 2015): a shared feature
extractor feeds a value stream V(s) and an advantage stream A(s, a), which
are combined as

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

The mean subtraction keeps the value and advantage streams identifiable.
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Dueling Q-network: state vector -> action values."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build the model.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): number of discrete actions
            seed (int): random seed
            fc1_units (int): number of nodes in the first hidden layer
            fc2_units (int): number of nodes in the second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2b = nn.Linear(fc2_units, 1)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.feature_stream = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )

        self.value_stream = nn.Sequential(
            self.fc2,
            nn.ReLU(),
            self.fc2b,
        )

        self.advantage_stream = nn.Sequential(
            self.fc2,
            nn.ReLU(),
            self.fc3,
        )

    def forward(self, state):
        """Map a batch of states to action values.

        Params
        ======
            state (torch.Tensor): batch of state vectors, shape (batch, state_size)

        Returns
        =======
            torch.Tensor: predicted action values, shape (batch, action_size)
        """
        features = self.feature_stream(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Dueling aggregation: the advantage mean is taken per sample over
        # actions, so each Q(s, a) depends only on its own sample's advantage.
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return qvals

import torch
import torch.nn as nn


ACTIVATION_FNS = {
    'nn.ELU': nn.ELU,
    'nn.ReLU': nn.ReLU,
    'nn.Tanh': nn.Tanh,
}

class ActorBC(nn.Module):
    def __init__(self, obs_dim, act_dim, net_arch_pi=[64, 64], activation_fn_str='nn.ELU'):
        super().__init__()
        if activation_fn_str not in ACTIVATION_FNS:
            raise ValueError(f"Unknown activation function: {activation_fn_str}")
        activation_fn = ACTIVATION_FNS[activation_fn_str]

        layers = []
        last = obs_dim
        for h in net_arch_pi:
            layers.append(nn.Linear(last, h))
            layers.append(activation_fn())
            last = h
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(last, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.policy_net(obs))
``` python

import torch
import numpy as np

class Agent(nn.Module):
    def __init__(
        self,
        state_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        conv_net: bool,
        joint_net: bool,
        device: str
    ):
        super(Agent, self).__init__()

        self.action_space = action_space
        self.state_space = state_space
        self.conv_net = conv_net
        self.joint_net = joint_net

        if len(state_space.shape) < 3:
            self.permute_states_fn = lambda x: x
        else:
            self.permute_states_fn = self.permute_states

        if conv_net:
            self.init_conv_net()
        else:
            self.init_dense_net()

        self.device = device
        self.to(self.device, dtype=torch.float32)

    def init_layer(self, layer, std: float=np.sqrt(2)):
        nn.init.orthogonal_(layer.weight, std)
        return layer

    def init_conv_net(self):
        state_shape = self.state_space.shape
        self.pi_backbone = nn.Sequential(
            self.init_layer(nn.Conv2d(state_shape[-1], 32, 8, stride=4)),
            nn.ReLU(),
            self.init_layer(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.init_layer(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.init_layer(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        if not self.joint_net:
            self.va_backbone = nn.Sequential(
                self.init_layer(nn.Conv2d(state_shape[-1], 32, 8, stride=4)),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                self.init_layer(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.policy = self.init_layer(nn.Linear(512, self.action_space.n), std=0.01)

        elif isinstance(self.action_space, gym.spaces.Box):
            self.policy = self.init_layer(nn.Linear(512, 2 * self.action_space.shape[0]), std=0.01)

        self.critic = self.init_layer(nn.Linear(512, 1), std=1.0)

    def init_dense_net(self):
        state_shape = self.state_space.shape
        self.pi_backbone = nn.Sequential(
            self.init_layer(nn.Linear(state_shape[0], 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh()
        )
        if not self.joint_net:
            self.va_backbone = nn.Sequential(
                self.init_layer(nn.Linear(state_shape[0], 64)),
                nn.Tanh(),
                self.init_layer(nn.Linear(64, 64)),
                nn.Tanh()
            )
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.policy = self.init_layer(nn.Linear(64, self.action_space.n), std=0.01)

        elif isinstance(self.action_space, gym.spaces.Box):
            self.policy = self.init_layer(nn.Linear(64, 2 * self.action_space.shape[0]), std=0.01)

        self.critic = self.init_layer(nn.Linear(64, 1), std=1.0)

    def permute_states(self, states: torch.Tensor):
        return torch.permute(states, (0, 3, 1, 2))

    def get_values(
        self,
        states: torch.Tensor
    ) -> torch.Tensor:

        states = self.permute_states_fn(states)

        if self.joint_net:
            hidden = self.pi_backbone(states)
            values = self.critic(hidden)
        else:
            hidden = self.va_backbone(states)
            values = self.critic(hidden)

        return torch.flatten(values)

    def get_actions_and_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        states = self.permute_states_fn(states)

        if self.joint_net:
            hidden = self.pi_backbone(states)
            policy_output = self.policy(hidden)
            values = self.critic(hidden)
        else:
            pi_hidden = self.pi_backbone(states)
            va_hidden = self.va_backbone(states)
            policy_output = self.policy(pi_hidden)
            values = self.critic(va_hidden)

        if isinstance(self.action_space, gym.spaces.Discrete):
            logits = policy_output
            action_dist = torch.distributions.Categorical(logits=logits)

            if actions is None:
                actions = action_dist.sample().to(dtype=torch.int32)

        elif isinstance(self.action_space, gym.spaces.Box):
            n = self.action_space.shape[0]

            lows = self.action_space.low
            highs = self.action_space.high
            if not isinstance(lows, np.ndarray):
                lows = lows * np.ones((n,), dtype=np.float32)
                highs = highs * np.ones((n,), dtype=np.float32)

            lows = torch.tensor(lows)
            highs = torch.tensor(highs)

            # Using a beta distribution with a, b >= 1 so it is unimodal
            modes = 0.5 * (torch.clip(policy_output[:, :n], min=-1.0, max=1.0) + 1) # Modes of beta distribution: (a - 1) / (a + b - 2)
            precisions = torch.exp(policy_output[:, n:])                            # Precisions of beta distribution: a + b - 2

            action_dist = ScaledBeta(modes, precisions, lows, highs)

            if actions is None:
                actions = action_dist.sample().to(dtype=torch.float32)

        log_probs = action_dist.log_prob(actions)
        values = values.flatten()
        entropy = action_dist.entropy().mean()

        return actions, log_probs, values, entropy

```
{: .scroll}

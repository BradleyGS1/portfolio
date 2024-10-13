```python

import torch
import numpy as np
import gymnasium as gym

from typing import Callable

class SyncVecEnv:
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        num_envs: int,
        steps_per_env: int,
        agent: Agent
    ):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.agent = agent

        self.rolling_ep_returns = [[] for _ in range(num_envs)]
        self.mean_ep_return = np.float32(np.nan)
        self.lower_ep_return = np.float32(np.nan)
        self.median_ep_return = np.float32(np.nan)
        self.upper_ep_return = np.float32(np.nan)

        self.rolling_ep_lengths = [[] for _ in range(num_envs)]
        self.mean_ep_length = np.float32(np.nan)

        state, _ = self.envs[0].reset()
        self.state_shape = state.shape
        self.action_space = self.envs[0].action_space
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dtype = torch.int32
        else:
            self.action_dtype = torch.float32

        if len(self.state_shape) < 3:
            self.permute_state_fn = lambda x: x
        else:
            self.permute_state_fn = self.permute_state
            new_state_shape = (self.state_shape[2], self.state_shape[0], self.state_shape[1])
            self.state_shape = new_state_shape

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_states = self.vec_reset()

    def permute_state(self, state: np.ndarray):
        return np.permute_dims(state, axes=(2, 0, 1))

    def close(self):
        self.envs = None

    def vec_reset(self) -> torch.Tensor:
        states = torch.zeros(size=(self.num_envs, *self.state_shape), dtype=torch.float32, device=self.device)
        for i, env in enumerate(self.envs):
            state = env.reset()[0]
            state = self.permute_state_fn(state)
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            states[i] = state
        return states

    def env_reset(self, env_id: int):
        env = self.envs[env_id]
        state = env.reset()[0]
        state = self.permute_state_fn(state)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return state

    def vec_step(
        self,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        actions = actions.cpu().numpy()

        states = torch.zeros(size=(self.num_envs, *self.state_shape), dtype=torch.float32, device=self.device)
        rewards = torch.zeros(size=(self.num_envs, ), dtype=torch.float32, device=self.device)
        done_flags = torch.zeros(size=(self.num_envs, ), dtype=torch.int32, device=self.device)
        trunc_flags = torch.zeros(size=(self.num_envs, ), dtype=torch.int32, device=self.device)
        for i, env in enumerate(self.envs):
            action = np.squeeze(actions[i])
            state, reward, done, trunc, info = env.step(action)
            state = self.permute_state_fn(state)
            states[i] = torch.tensor(state, dtype=torch.float32, device=self.device)
            rewards[i] = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done_flags[i] = torch.tensor(done, dtype=torch.int32, device=self.device)
            trunc_flags[i] = torch.tensor(trunc, dtype=torch.int32, device=self.device)

            if "episode" in info:
                ep_return = info["episode"]["r"].item()
                roll_ep_returns = self.rolling_ep_returns[i]

                ep_length = info["episode"]["l"].item()
                roll_ep_lengths = self.rolling_ep_lengths[i]

                if len(roll_ep_returns) == 10:
                    roll_ep_returns.pop(0)
                roll_ep_returns.append(ep_return)

                if len(roll_ep_lengths) == 10:
                    roll_ep_lengths.pop(0)
                roll_ep_lengths.append(ep_length)

        return states, rewards, done_flags, trunc_flags

    def test_agent(self, render_every: int):
        num_steps = self.steps_per_env
        num_envs = self.num_envs
        self.images = [[] for _ in range(num_envs)]

        t_states = self.vec_reset()
        pbar = tqdm.trange(num_steps)
        pbar.set_description_str("Recording")
        for t_step in pbar:

            if t_step % render_every == 0:
                for actor in range(num_envs):
                    state_image = Image.fromarray(self.envs[actor].render(), mode="RGB")
                    self.images[actor].append(state_image)

            t_actions, _, _, _ = self.agent.get_actions_and_values(t_states, actions=None)
            t_states, _, _, _ = self.vec_step(t_actions)

        self.vec_reset()

    def rollout(self):
        with torch.no_grad():
            num_steps = self.steps_per_env
            num_envs = self.num_envs

            self.states = torch.zeros(size=(num_steps, num_envs, *self.state_shape), dtype=torch.float32, device=self.device)
            self.actions = torch.zeros(size=(num_steps, num_envs, ), dtype=self.action_dtype, device=self.device)
            self.rewards = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)
            self.done_flags = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.int32, device=self.device)
            self.trunc_flags = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.int32, device=self.device)
            self.values = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)
            self.log_probs = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)

            end_states = [[] for _ in range(num_envs)]
            self.ep_counts = torch.zeros(size=(self.num_envs, ), dtype=torch.int32, device=self.device)
            self.total_return = np.float32(0.0)

            for t_step in range(num_steps):

                # Compute actions, log_probs and critic values for each env using their states
                t_actions, t_log_probs, t_values, _  = self.agent.get_actions_and_values(self.t_states, actions=None)

                # Perform a vector env step using the sampled actions
                t_new_states, t_rewards, t_dones, t_truncs = self.vec_step(t_actions)

                for actor in range(self.num_envs):
                    reward = t_rewards[actor]
                    done = t_dones[actor]
                    trunc = t_truncs[actor]
                    terminated = done + trunc

                    self.total_return += reward.cpu().numpy()

                    can_reset = True
                    if terminated == 0 and t_step == num_steps - 1:
                        terminated += 1
                        t_truncs[actor] += 1
                        can_reset = False

                    if terminated > 0:
                        end_state = t_new_states[actor]
                        end_states[actor].append(end_state.cpu())

                        if can_reset:
                            t_new_states[actor] = self.env_reset(actor)

                        self.ep_counts[actor] += 1

                self.states[t_step] = self.t_states
                self.actions[t_step] = t_actions
                self.rewards[t_step] = t_rewards
                self.done_flags[t_step] = t_dones
                self.trunc_flags[t_step] = t_truncs
                self.values[t_step] = t_values
                self.log_probs[t_step] = t_log_probs

                self.t_states = t_new_states

        self.states.requires_grad = True
        end_states_tensors = [torch.stack(actor_end_states, dim=0) for actor_end_states in end_states]
        self.end_states = torch.concatenate(end_states_tensors, dim=0).to(device=self.device)

        ep_returns_stack = np.concatenate(self.rolling_ep_returns)
        if len(ep_returns_stack) > 0:
            self.mean_ep_return = np.mean(ep_returns_stack, dtype=np.float32)
            self.lower_ep_return, self.median_ep_return, self.upper_ep_return = np.percentile(
                ep_returns_stack, [5.0, 50.0, 95.0])

        ep_lengths_stack = np.concatenate(self.rolling_ep_lengths)
        if len(ep_lengths_stack) > 0:
            self.mean_ep_length = np.mean(ep_lengths_stack, dtype=np.float32)

```
{: .scroll}
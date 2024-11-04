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
        render_every: int,
        render_fps: float,
        agent: Agent
    ):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.global_steps = 0

        self.render_every = (render_every if render_every > 0 else 1)
        self.can_record = render_every > 0
        self.ready_to_record = False
        self.is_recording = self.can_record
        self.record_episode = 0
        self.record_buffer = []
        self.record_total_reward = 0.0
        self.render_fps = render_fps
        self.render_folder = "./renders/misc"
        if self.can_record and wandb.run is not None:
            project_name = wandb.run.project
            run_name = wandb.run.name
            self.render_folder = f"./renders/{project_name}/{run_name}"
        os.makedirs(self.render_folder, exist_ok=True)

        self.agent = agent

        self.max_ep_return = np.float32(np.nan)
        self.lower_ep_return = np.float32(np.nan)
        self.median_ep_return = np.float32(np.nan)
        self.upper_ep_return = np.float32(np.nan)
        self.median_ep_length = np.float32(np.nan)

        self.state_space = agent.state_space
        self.action_space = agent.action_space
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dtype = torch.int32
        else:
            self.action_dtype = torch.float32

        self.device = self.agent.device
        self.t_states = self.vec_reset()

    def close(self):
        self.envs = None

    def vec_reset(self) -> torch.Tensor:
        states = torch.zeros(size=(self.num_envs, *self.state_space.shape), dtype=torch.float32, device=self.device)
        for i, env in enumerate(self.envs):
            state = env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            states[i] = state
        return states

    def env_reset(self, env_id: int):
        env = self.envs[env_id]
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return state

    def vec_step(
        self,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        actions = actions.cpu().numpy()

        states = torch.zeros(size=(self.num_envs, *self.state_space.shape), dtype=torch.float32, device=self.device)
        rewards = torch.zeros(size=(self.num_envs, ), dtype=torch.float32, device=self.device)
        done_flags = torch.zeros(size=(self.num_envs, ), dtype=torch.int32, device=self.device)
        trunc_flags = torch.zeros(size=(self.num_envs, ), dtype=torch.int32, device=self.device)
        for i, env in enumerate(self.envs):
            action = np.squeeze(actions[i])
            state, reward, done, trunc, info = env.step(action)
            states[i] = torch.tensor(state, dtype=torch.float32, device=self.device)
            rewards[i] = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done_flags[i] = torch.tensor(done, dtype=torch.int32, device=self.device)
            trunc_flags[i] = torch.tensor(trunc, dtype=torch.int32, device=self.device)

        return states, rewards, done_flags, trunc_flags

    def rollout(self):
        with torch.no_grad():
            num_steps = self.steps_per_env
            num_envs = self.num_envs

            self.states = torch.zeros(size=(num_steps, num_envs, *self.state_space.shape), dtype=torch.float32, device=self.device)
            self.actions = torch.zeros(size=(num_steps, num_envs, *self.action_space.shape), dtype=self.action_dtype, device=self.device)
            self.rewards = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)
            self.done_flags = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.int32, device=self.device)
            self.trunc_flags = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.int32, device=self.device)
            self.values = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)
            self.log_probs = torch.zeros(size=(num_steps, num_envs, ), dtype=torch.float32, device=self.device)

            end_states = [[] for _ in range(num_envs)]
            self.total_return = np.float32(0.0)

            for t_step in range(num_steps):

                # Record rendered observation from the first environment if recording
                if self.is_recording:
                    obs_render = self.envs[0].render().astype(np.uint8)
                    obs_image = Image.fromarray(obs_render)

                    draw = ImageDraw.Draw(obs_image)
                    font = ImageFont.load_default()
                    text = f"Total Reward: {self.record_total_reward}"
                    position = (50, 40)
                    text_color = (0, 204, 102)
                    draw.text(position, text, text_color, font)

                    self.record_buffer.append(obs_image) 

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
                    if actor == 0:
                        self.record_total_reward += reward.cpu().numpy()

                    can_reset = True
                    if terminated == 0 and t_step == num_steps - 1:
                        terminated += 1
                        t_truncs[actor] += 1
                        can_reset = False

                    elif terminated > 0 and actor == 0:
                        if self.is_recording:
                            self.is_recording = False
                            if len(self.record_buffer) > 1:
                                self.record_buffer[0].save(
                                        f"{self.render_folder}/render_{self.record_episode}.gif", 
                                        save_all=True, 
                                        append_images=self.record_buffer[1:], 
                                        duration=1000/self.render_fps, 
                                        loop=0)

                            self.record_buffer = []
                            self.record_episode += 1

                        elif self.ready_to_record:
                            self.ready_to_record = False
                            self.is_recording = True
                            self.record_total_reward = 0.0

                    if terminated > 0:
                        end_state = t_new_states[actor]
                        end_states[actor].append(end_state.cpu())

                        if can_reset:
                            t_new_states[actor] = self.env_reset(actor)

                    self.global_steps += 1

                    ready_to_record = self.global_steps % self.render_every == self.render_every - 1
                    if self.can_record and ready_to_record:
                        self.ready_to_record = True

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

        if not hasattr(self.envs[0], "return_queue"):
            return

        ep_returns_stack = np.concatenate([np.array(env.return_queue).reshape(-1) for env in self.envs])
        if len(ep_returns_stack) > 0:
            if np.isnan(self.max_ep_return) or np.max(ep_returns_stack) > self.max_ep_return:
                self.max_ep_return = np.max(ep_returns_stack)

            self.lower_ep_return, self.median_ep_return, self.upper_ep_return = np.percentile(
                ep_returns_stack, [5.0, 50.0, 95.0])

        ep_lengths_stack = np.concatenate([np.array(env.length_queue).reshape(-1) for env in self.envs])
        if len(ep_lengths_stack) > 0:
            self.median_ep_length = np.percentile(ep_lengths_stack, 50.0)

```
{: .scroll}

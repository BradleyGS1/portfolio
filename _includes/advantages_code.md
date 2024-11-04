``` python

import torch

def compute_advantages(
	self,
	rewards: torch.Tensor,
	values: torch.Tensor,
	end_values: torch.Tensor,
	done_flags: torch.Tensor,
	trunc_flags: torch.Tensor
	):

	num_steps = rewards.size(0)
	num_agents = rewards.size(1)

	advantages = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)

	discount_factor = self.discount_factor
	gae_factor = self.gae_factor

	ep_counts = torch.sum(done_flags + trunc_flags - done_flags * trunc_flags, dim=0, dtype=torch.int32)
	end_indices = torch.cumsum(ep_counts, dim=0) - 1
	next_values = torch.zeros(size=(num_agents,), dtype=torch.float32, device=self.device)
	next_advantages = torch.zeros(size=(num_agents,), dtype=torch.float32, device=self.device)

	for t in reversed(range(num_steps)):
	dones = done_flags[t]
	truncs = trunc_flags[t]
	terminations = dones + truncs

	next_values = (1 - terminations) * next_values + truncs * end_values[end_indices]
	next_advantages = (1 - terminations) * next_advantages
	end_indices = end_indices - terminations

	td_residuals = rewards[t] + discount_factor * next_values - values[t]
	advantages[t] = td_residuals + discount_factor * gae_factor * next_advantages

	next_values = values[t]
	next_advantages = advantages[t]

	return advantages

```
{: .scroll}

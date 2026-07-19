import torch as T


def learn(model, model_opt, saved_experiences, next_value, gamma=0.99, grad_norm=0.5):
    memory_observations, memory_actions, memory_logprobs, memory_values, memory_rewards = saved_experiences
    saved_logprobs = memory_logprobs
    saved_states = memory_observations
    saved_rewards = [
        reward.detach().item()
        if isinstance(reward, T.Tensor)
        else float(reward)
        for reward in memory_rewards
    ]
    device = next(model.parameters()).device
    #prepare returns
    R = next_value.detach().item()
    returns = [0 for _ in range(len(saved_logprobs))]
    critic_vals = [0. for _ in range(len(saved_logprobs))]
    for i in range(1, len(returns)+1):
        R = saved_rewards[-i] + gamma*R
        returns[-i]=R
        state = T.as_tensor(saved_states[-i], dtype=T.float32, device=device)
        if state.ndim == 2:
            state = state.unsqueeze(0)
        critic_vals[-i] = model.critic(state).squeeze()
    saved_logprobs = T.stack(saved_logprobs)
    critic_vals = T.stack(critic_vals)
    returns = T.tensor(returns, dtype=T.float32, device=device)
    advantage = (returns - critic_vals).detach()
    #update actor
    agent_loss = -(saved_logprobs*advantage).sum()

    #update critic
    critic_loss = (returns-critic_vals)**2
    critic_loss = critic_loss.mean()

    loss = agent_loss + critic_loss
    model_opt.zero_grad(set_to_none=True)
    loss.backward()
    T.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm)
    model_opt.step()
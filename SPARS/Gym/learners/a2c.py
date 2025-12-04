import torch as T

def evaluate(model, batch_obs, batch_acts):
    logits, V = model(batch_obs)
    
    dist = T.distributions.Normal(logits, 0.02)
    log_probs = dist.log_prob(batch_acts)
    
    return V, log_probs

def learn(model, model_opt, saved_experiences,
          gamma: float = 0.99):
    memory_actions, memory_features, memory_rewards = saved_experiences
    memory_features = T.stack(memory_features, dim=0)
    memory_actions = T.stack(memory_actions, dim=0)
    memory_rewards = T.stack(memory_rewards, dim=0)

    device = model.device
    rews = T.stack([r.to(device).float().view(-1).mean() if isinstance(r, T.Tensor)
                    else T.tensor(float(r), device=device)
                    for r in memory_rewards])
    device = next(model.parameters()).device

    Tlen = len(memory_rewards)

    value, batch_logprob  = evaluate(model, memory_features, memory_actions)
    batch_logprob = batch_logprob.detach()
    value = value

    delta = T.zeros(Tlen, dtype=value.dtype, device=value.device)

    delta[Tlen-1] = rews[Tlen-1]

    for t in range(Tlen-2, -1, -1):
        delta[t] = rews[t] + gamma * value[t + 1] - value[t]

    advantages = T.zeros(Tlen, dtype=value.dtype, device=value.device)

    curr_advantage = delta[Tlen-1]

    for t in range(Tlen-2, -1, -1):
        curr_advantage = delta[t] + gamma * curr_advantage
        advantages[t] = curr_advantage

    updates_per_iterations = 1
    clip = 0.2
    for i in range(updates_per_iterations):
        _, current_logprob = evaluate(model, memory_features, memory_actions)
        ratio = T.exp(current_logprob - batch_logprob)
        advantages = advantages.view(-1, 1, 1, 1)
        surr1 = advantages * ratio
        surr2 = T.clamp(ratio, 1-clip, 1+clip) * advantages
        
        actor_loss = -T.min(surr1, surr2).mean() 
                
        model_opt.zero_grad()
        actor_loss.backward()
        model_opt.step()
        

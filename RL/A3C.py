import logging
import random
from typing import Iterator

import gym
import numpy as np
import torch
from gym.spaces import Box
from torch.nn import Parameter

logger = torch.multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)


class PolicyValueModel(torch.nn.Module):

    def __init__(self, state_size, action_size, hidden_size, is_continuous=False):
        super().__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size),
            torch.nn.SiLU(),
        )
        self.is_continuous = is_continuous
        if is_continuous:
            self.policy_nn = torch.nn.Linear(hidden_size, action_size)
            self.log_std = torch.nn.parameter.Parameter(-0.5 * torch.ones(action_size))
        else:
            self.policy_nn = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, action_size),
                torch.nn.Softmax(dim=0)
            )
        self.value_nn = torch.nn.Linear(hidden_size, 1)

    def forward(self, state):
        if type(state) is not torch.Tensor:
            state = torch.tensor(state)
        hidden = self.nn(state)
        if self.is_continuous:
            mu = self.policy_nn(hidden)
            action_dist = torch.distributions.Normal(mu, torch.exp(self.log_std))
        else:
            action_dist = self.policy_nn(hidden)
            action_dist = torch.distributions.Categorical(action_dist)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        action = action.detach().numpy()

        return action_dist, action, action_log_prob, self.value_nn(hidden)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params: Iterator[Parameter], **kwargs):
        super().__init__(params, **kwargs)
        for pg in self.param_groups:
            for p in pg['params']:
                self.state[p]['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]['exp_avg'].share_memory_()
                self.state[p]['exp_avg_sq'].share_memory_()
                self.state[p]['step'] = 0


class A3CAgent(torch.multiprocessing.Process):

    def __init__(self,
                 name,
                 env_name,
                 capacity,
                 global_model,
                 optimizer,
                 max_episodes,
                 train_every_step,
                 early_stopping_reward,
                 gamma=0.9,
                 entropy_loss_weight=0.01,
                 max_episode_length=500,
                 verbose=False,
                 terminate_flag=None):
        super().__init__()
        self.name = name
        self.env_name = env_name
        self.capacity = capacity
        self.global_model = global_model
        self.optimizer = optimizer
        self.max_episodes = max_episodes
        self.train_every_step = train_every_step
        self.early_stopping_reward = early_stopping_reward
        self.gamma = gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.max_episode_length = max_episode_length
        self.verbose = verbose
        self.terminate_flag = terminate_flag

    def run(self):
        logger.info(f'Agent {self.name} starting')
        with gym.make(self.env_name) as env:
            n = env.action_space.shape[0] if type(env.action_space) is Box else env.action_space.n
            model = PolicyValueModel(env.observation_space.shape[0], n, self.capacity, type(env.action_space) is Box)
            model.train()
            global_step = 0
            losses = []
            loss_factors = []
            episode_lengths = []
            total_rewards = []
            entropies = []
            for episode in range(self.max_episodes):
                state = env.reset()
                total_rewards.append(0)
                rollout = []
                for step in range(self.max_episode_length):
                    action_dist, action, action_log_prob, value = model.forward(state)
                    new_state, reward, done, _ = env.step(action)
                    if len(rollout) > 0:
                        rollout[-1]['new_state_value'] = value

                    total_rewards[-1] += reward
                    rollout.append({
                        'state': state,
                        'state_value': value,
                        'action_dist': action_dist,
                        'action': action,
                        'action_log_prob': action_log_prob,
                        'reward': reward,
                        'new_state': new_state,
                        'done': done
                    })

                    state = new_state
                    if self.terminate_flag is not None and self.terminate_flag.value > 0:
                        logger.info('Early exit due to terminate_flag == 1')
                        break

                    if len(rollout) == self.train_every_step or done:
                        # ******************************** Training step **************************************
                        if not done:
                            rollout[-1]['new_state_value'] = model.forward(rollout[-1]['new_state'])[3].item()
                        else:
                            rollout[-1]['new_state_value'] = 0

                        # Calculate n-step returns
                        returns = []
                        total_return = rollout[-1]['new_state_value']
                        for t in rollout[::-1]:
                            returns.append(t['reward'] + self.gamma * total_return)
                        returns = returns[::-1]

                        # Calculate advantages. E(Q_t) - E(V_t) ~ R_{t+1} + gamma*E(V_{t+1}) - E(V_t)
                        advantages = []
                        for t, r in zip(rollout, returns):
                            advantages.append(r - t['state_value'].item())
                        advantages = torch.as_tensor(advantages)

                        # value update MSE error for gradient **descent** = ({t['reward'] + gamma * t['new_state_value']} - t['state_value'])^2
                        #                       grad(value) = ({t['reward'] + gamma * t['new_state_value']} - t['state_value'])*grad(-t['state_value'])
                        #                                   = -advantage * grad(t['state_value'])
                        values = torch.cat([t['state_value'] for t in rollout])
                        value_loss = -(advantages * values).mean()

                        # policy update for gradient **ascent** = -advantage * log(t['action_dist'][t['action']]))
                        action_probs = torch.stack([t['action_log_prob'] for t in rollout])
                        if len(action_probs.shape) == 2:
                            action_probs = torch.sum(action_probs, dim=1)
                        policy_loss = -(advantages * action_probs).mean()

                        # policy entropy regularization loss = beta * (-H(action_dist))
                        entropy = torch.stack([t['action_dist'].entropy() for t in rollout])
                        entropy = entropy.mean()
                        entropy_loss = -self.entropy_loss_weight * entropy
                        entropies.append(entropy.item())

                        loss = value_loss + policy_loss + entropy_loss
                        self.optimizer.zero_grad()
                        loss.backward()
                        for pg, pl in zip(self.global_model.parameters(), model.parameters()):
                            pg.grad = pl.grad
                        self.optimizer.step()
                        model.load_state_dict(self.global_model.state_dict())

                        losses.append(loss.item())
                        loss_factors.append((value_loss.item(), policy_loss.item(), entropy_loss.item()))

                        rollout = []

                    if done:
                        episode_lengths.append(step)
                        break
                    global_step += 1

                if self.terminate_flag is not None and self.terminate_flag.value > 0:
                    break

                if np.mean(total_rewards[-10:]) > self.early_stopping_reward:
                    logger.info(f'{self.name} Total rewards > {self.early_stopping_reward}, early stopping!')
                    if self.terminate_flag is not None:
                        self.terminate_flag.value = 1
                    break

                if random.random() < 0.05:
                    logger.info(f'{self.name} episode:{episode}; '
                                f'loss:{np.mean(losses[-10:]):06.3f}; '
                                f'episode len:{np.mean(episode_lengths[-10:]):06.0f}; '
                                f'total reward:{np.mean(total_rewards[-10:]):06.0f} '
                                f'entropy:{np.mean(entropies[-10:]):06.3f} ')

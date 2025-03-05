import torch
from torch import nn
import gymnasium as gym
from dataclasses import dataclass, field
from torch.distributions.categorical import Categorical
import numpy as np
import time

@dataclass
class Args:
    gamma: float = 0.99
    lambda_gae: float = 0.95
    epsilon_clip: float = 0.2
    num_steps: int = 2048
    batch_size: int = 64
    epochs: int = 10
    lr: float = 2.5e-4
    
@dataclass
class MetricsContainer:
    num_steps:int
    episode_rewards:list[float] = field(default_factory=list)
    actor_losses:list[float] = field(default_factory=list)
    critic_losses:list[float] = field(default_factory=list)
    entropy_losses:list[float] = field(default_factory=list)
    entropy_mean:list[float] = field(default_factory=list)
    
    def __post_init__(self):
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.num_updates = 1
        
    def reset(self):
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.entropy_mean = []
        self.last_iteration_time = time.time()
        self.num_updates += 1
        
    def show(self):
        print(f'num_steps_global : {self.num_updates*self.num_steps}')
        print(f'time_elapsed : {time.time() - self.start_time}')
        print(f'step/seconds : {self.num_steps/(time.time() - self.last_iteration_time)}')
        print(f'len_mean : {self.num_steps/len(self.episode_rewards)}')
        print(f'mean_reward : {np.mean(self.episode_rewards)}')
        print(f'mean_actor_loss : {np.mean(self.actor_losses)}')
        print(f'mean_critic_loss : {np.mean(self.critic_losses)}')
        print(f'mean_entropy : {np.mean(self.entropy_mean)}')
        print(f'\n')
        self.reset()

@dataclass
class ExperienceBuffer:
    def __init__(self, ppo):
        self.ppo = ppo
        self.num_steps = ppo.args.num_steps
        env = ppo.env
        self.cursor = 0
        self.states = torch.zeros(self.num_steps, env.observation_space.shape[0])
        self.actions = torch.zeros(self.num_steps, 1)
        self.rewards = torch.zeros(self.num_steps, 1)
        self.next_states = torch.zeros(self.num_steps, env.observation_space.shape[0])
        self.logprobs = torch.zeros(self.num_steps, 1)
        self.dones = torch.zeros(self.num_steps, 1)
        self.advantages = torch.zeros(self.num_steps, 1)
        self.returns = torch.zeros(self.num_steps, 1)
    
    def append(self, state, action, reward, next_state, logprobs, done):
        self.states[self.cursor] = state
        self.actions[self.cursor] = action
        self.rewards[self.cursor] = reward
        self.next_states[self.cursor] = next_state
        self.logprobs[self.cursor] = logprobs
        self.dones[self.cursor] = done
        self.cursor += 1
        
    def compute_advantages_and_returns(self):
        with torch.no_grad():
            values = self.ppo.get_value(self.states)
            next_value = self.ppo.get_value(self.next_states[-1]).reshape(1, -1)  # Dernière valeur de l’épisode

            advantages = torch.zeros_like(self.rewards).to(self.states.device)  # Initialisation propre
            lastgaelam = 0  # Stocke l’avantage précédent pour propagation récursive

            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.dones[-1]  # Vérifie si l’épisode est réellement terminé
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]  # 0 si épisode terminé, 1 sinon
                    nextvalues = values[t + 1]  # Utilisation des valeurs futures

                delta = self.rewards[t] + self.ppo.args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + self.ppo.args.gamma * self.ppo.args.lambda_gae * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam

            self.returns = advantages + values  # Ajout des valeurs pour obtenir les retours
            self.advantages = advantages
            
    def reset(self):
        self.cursor = 0
        
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOTorch(nn.Module):
    def __init__(self, env, args):
        super(PPOTorch, self).__init__()
        
        # Initialization of env
        self.env = env
        
        # Initialization of ppo variable
        self.args = args
        self.actual_step = 1
        
        # Initialization of actor and critic networks and their optimizers
        self.actor = self.create_network('actor')
        self.critic = self.create_network('critic')
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        
        # Variable used in the algorithm
        self.buffer = ExperienceBuffer(self)
        # Variable used in metrics
        self.metrics = MetricsContainer(args.num_steps)
        
    def create_network(self, net_type):
        return nn.Sequential(
            layer_init(nn.Linear(self.env.observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.env.action_space.n if net_type == 'actor' else 1), std=0.01 if net_type == 'actor' else 1),
        )
        
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def get_value(self, x):
        return self.critic(x)
    
    def train(self, total_timesteps):
        while self.actual_step < total_timesteps:
            state, _ = self.env.reset(seed=12)
            state = torch.tensor(state, dtype=torch.float32)
            episode_reward = 0
            done = False
            entropies = []
            while not done:
                with torch.no_grad():
                    action, logprobs, entropy, _ = self.get_action_and_value(state)
                entropies.append(entropy)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                next_state, reward = torch.tensor(next_state, dtype=torch.float32), torch.tensor(reward, dtype=torch.float32)
                episode_reward += reward.item()
                
                # I have decided to use a class Experience to make the code easier to understand
                self.buffer.append(state, action, reward, next_state, logprobs, done)
                if self.actual_step % self.args.num_steps == 0 and self.actual_step != 0:
                    self.update()
                    self.metrics.show()

                state = next_state
                self.actual_step += 1
            self.metrics.episode_rewards.append(episode_reward)
            self.metrics.entropy_mean.append(torch.stack(entropies).mean().item())
                
    def update(self):
        self.buffer.compute_advantages_and_returns()
        for _ in range(self.args.epochs):
            self.update_actor_policy()
            self.update_critic()
        self.buffer.reset()
            
    def update_actor_policy(self):
        def compute_clip_loss(old_policies, new_policies, advantages):
            logratios = new_policies - old_policies
            ratios = torch.exp(logratios)
            clipped_ratios = torch.clamp(ratios, 1 - self.args.epsilon_clip, 1 + self.args.epsilon_clip)
            loss = -torch.min(ratios * advantages, clipped_ratios * advantages)
            return loss.mean()
        
        old_policies = self.buffer.logprobs
        states = self.buffer.states
        advantages = self.buffer.advantages
        actions = self.buffer.actions
        
        global_indices = torch.randperm(self.args.num_steps)
        for start in range(0, self.args.num_steps, self.args.batch_size):
            end = start + self.args.batch_size
            local_indices = global_indices[start:end]
            batch_states = states[local_indices]
            batch_actions = actions[local_indices]
            batch_old_policies = old_policies[local_indices]
            batch_advantages = advantages[local_indices]
            
            _, batch_new_policies, _, _ = self.get_action_and_value(batch_states, batch_actions)
        
            loss = compute_clip_loss(batch_old_policies, batch_new_policies, batch_advantages)
            
            # Metrics
            self.metrics.actor_losses.append(loss.item())
            
            self.optimizer_actor.zero_grad()
            loss.backward()
            self.optimizer_actor.step()
            
    def update_critic(self):
        def compute_loss_critic(states, returns):
            return 0.5 * ((self.critic(states) - returns) ** 2).mean()
        
        states = self.buffer.states
        returns = self.buffer.returns
        
        global_indices = torch.randperm(self.args.num_steps)
        for start in range(0, self.args.num_steps, self.args.batch_size):
            end = start + self.args.batch_size
            local_indices = global_indices[start:end]
            batch_states = states[local_indices]
            batch_targets = returns[local_indices]
            
            loss = compute_loss_critic(batch_states, batch_targets)
            
            # Metrics
            self.metrics.critic_losses.append(loss.item())
            
            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_critic.step()
    
if __name__ == '__main__':
    args = Args()
    env = gym.make('CartPole-v1')
    if True:
        ppo = PPOTorch(env, args)
        ppo.train(10000000)
    else:
        from stable_baselines3 import PPO
        model = PPO('MlpPolicy', env, verbose=1, batch_size=4096, n_steps=4096)
        model.learn(total_timesteps=1000000)
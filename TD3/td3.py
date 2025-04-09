import torch
from torch import nn
import gymnasium as gym
from dataclasses import dataclass, field
from torch.distributions.categorical import Categorical
import numpy as np
import time
from tabulate import tabulate
import random 

@dataclass
class Args:
    seed: int | None = 2
    gamma: float = 0.99 
    learning_rate: float = 3e-4 # Can be tuned
    buffer_size: int = int(1e6)
    tau: float = 0.005
    batch_size: int = 256
    exploration_noise: float = 0.1
    learning_starts: int = 100
    policy_frequency: int = 2
    noise_clip: float = 0.5
    show_frequency: int = 1000

@dataclass
class ReplayBuffer:
    size: int
    obs_dim: int
    action_dim: int
    
    def __post_init__(self):
        self.cursor = 0
        self.obs_buf = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.size, self.action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((self.size,), dtype=np.float32)
        self.done_buf = np.zeros((self.size,), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        
    def add(self, obs, action, reward, done, next_obs):
        self.obs_buf[self.cursor] = obs
        self.action_buf[self.cursor] = action
        self.reward_buf[self.cursor] = reward
        self.done_buf[self.cursor] = done
        self.next_obs_buf[self.cursor] = next_obs
        
        self.cursor = (self.cursor + 1) % self.size
        if self.size < len(self.obs_buf):
            self.size += 1
            
    def sample(self, batch_size : int):
        assert batch_size <= self.size, "Batch size must be less than the size of the buffer."
        assert type(batch_size) == int, "Batch size must be an integer."
        
        idx = np.random.randint(0, self.size, size=batch_size)
        obs_batch = torch.FloatTensor(self.obs_buf[idx])
        action_batch = torch.FloatTensor(self.action_buf[idx])
        reward_batch = torch.FloatTensor(self.reward_buf[idx])
        done_batch = torch.FloatTensor(self.done_buf[idx])
        next_obs_batch = torch.FloatTensor(self.next_obs_buf[idx])
        
        return obs_batch, action_batch, reward_batch, done_batch, next_obs_batch
            
@dataclass
class MetricsContainer:
    num_steps:int
    episode_rewards:list[float] = field(default_factory=list)
    episode_length:list[float] = field(default_factory=list)
    critic_losses:list[float] = field(default_factory=list)
    actor_losses:list[float] = field(default_factory=list)
    
    def __post_init__(self):
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.num_updates = 1
        
    def reset(self):
        self.episode_rewards = []
        self.episode_length = []
        self.critic_losses = []
        self.actor_losses = []
        self.last_iteration_time = time.time()
        self.num_updates += 1
        
    def show(self):
        datas = [
            ["Number of timesteps", f"{self.num_updates * self.num_steps:.0f}"],
            ["Number of updates", f"{self.num_updates * 10:.0f}"],
            ["Mean reward", f"{np.mean(self.episode_rewards):.2f}"],
            ["Mean episode length", f"{np.mean(self.episode_length):.2f}"],
            ["Time elapsed", f"{(time.time() - self.start_time):.2f} s"],
            ["Time per update", f"{(time.time() - self.last_iteration_time) / self.num_updates:.2f} s"],
            ["Steps per second", f"{self.num_steps / (time.time() - self.last_iteration_time):.2f}"],
            ["Critic loss", f"{np.mean(self.critic_losses):.2f}"],
            ["Actor loss", f"{np.mean(self.actor_losses):.2f}"],
        ]
        
        print(tabulate(datas, headers=["Metrics", "Value"], tablefmt="fancy_grid"))
        self.reset()

class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0] + env.action_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.shape[0])
        
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_scale + self.action_bias
    
class TD3Trainer:
    def __init__(self, env, args, device: torch.device | None = None):
        self.env = env
        self.args = args
        
        self.device = device if device is not None else torch.device('cpu')
        
        if self.args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            self.env.action_space.seed(args.seed)
            self.env.observation_space.seed(args.seed)
            
        # Initialize actor and critic networks
        self.actor = Actor(env).to(self.device)
        self.qnetwork1 = QNetwork(env).to(self.device)
        self.qnetwork2 = QNetwork(env).to(self.device)
        
        # Initialize target networks
        self.target_actor = Actor(env).to(torch.device(self.device))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_q_network1 = QNetwork(env).to(torch.device(self.device))
        self.target_q_network1.load_state_dict(self.qnetwork1.state_dict())
        self.target_q_network2 = QNetwork(env).to(torch.device(self.device))
        self.target_q_network2.load_state_dict(self.qnetwork2.state_dict())
        
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(list(self.qnetwork1.parameters()) + list(self.qnetwork2.parameters()), lr=args.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.qnetwork1.parameters(), lr=args.learning_rate)
        
        self.replay_buffer = ReplayBuffer(args.buffer_size, env.observation_space.shape[0], env.action_space.shape[0])
        self.metrics = MetricsContainer(args.batch_size)
        
    def learn(self, num_timesteps: int = 10000):
        obs, _ = self.env.reset(seed=self.args.seed)
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_timesteps):
            if step < self.args.learning_starts:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    action = self.actor(torch.FloatTensor(obs).to(self.device))
                    # Add exploration noise
                    action += torch.normal(0, self.actor.action_scale * self.args.exploration_noise)
                    # Put the variable in the cpu
                    action = action.cpu().numpy()
                    # Clip action to valid range
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                    
            next_obs, reward, done, truncation, _ = self.env.step(action)
            self.replay_buffer.add(obs, action, reward, done, next_obs)
            obs = next_obs
            
            if step >= self.args.learning_starts:
                self.train()
                
            # Metrics
            episode_reward += reward
            episode_length += 1
            if done or truncation:
                self.metrics.episode_rewards.append(episode_reward)
                self.metrics.episode_length.append(episode_length)
                episode_reward = 0
                episode_length = 0
                obs, _ = self.env.reset(seed=self.args.seed)
                
            if step % self.args.show_frequency == 0 and step > self.args.learning_starts:
                self.metrics.show()
                
    def train(self):
        if self.replay_buffer.size < self.args.batch_size:
            return
        
        # Sample a batch
        obs_b, action_b, reward_b, done_b, next_obs_b = self.replay_buffer.sample(self.args.batch_size)
        
        # Update critic
        critic_loss = self.update_critic(obs_b, action_b, reward_b, done_b, next_obs_b)
        
        # Update actor
        actor_loss = self.update_actor(obs_b)
        critic_loss, actor_loss 
        self.metrics.critic_losses.append(critic_loss)
        self.metrics.actor_losses.append(actor_loss)
        
    def update_critic(self, obs_b, action_b, reward_b, done_b, next_obs_b):
        with torch.no_grad():    
            # Predict the next action using the target actor
            next_action = self.target_actor(next_obs_b)
            
            # TD3 improvement: add noise to the next action
            # We apply a noise to regularize the qnetwork and try to avoid overestimation bias
            clipped_noise = (torch.randn_like(action_b) * self.args.exploration_noise).clamp(-self.args.noise_clip, self.args.noise_clip) * self.target_actor.action_scale
            next_action_with_noise = (next_action + clipped_noise).clamp(self.env.action_space.low[0], self.env.action_space.high[0])
            
            # Compute the target Q value using the target Q network and the next action for the twins qnetwork
            # Formula: Q(s,a) = r + γ * Q(s', a')
            target_q1 = self.target_q_network1(next_obs_b, next_action_with_noise).squeeze(1)
            target_q2 = self.target_q_network2(next_obs_b, next_action_with_noise).squeeze(1)
            min_target_q = torch.min(target_q1, target_q2)
            next_q_value = reward_b + (1 - done_b) * self.args.gamma * min_target_q
                        
        loss_q1 = nn.MSELoss()(self.qnetwork1(obs_b, action_b).squeeze(), next_q_value)
        loss_q2 = nn.MSELoss()(self.qnetwork2(obs_b, action_b).squeeze(), next_q_value)
        loss = loss_q1 + loss_q2
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return loss.item()
    
    def update_actor(self, obs_b):
        # Compute the actor loss using the Q network
        # Formula: J(θ) = -E[Q(s, μ(s|θ))]
        actor_loss = -self.qnetwork1(obs_b.to(self.device), self.actor(obs_b.to(self.device))).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_q_network1, self.qnetwork1)
        self.soft_update(self.target_q_network2, self.qnetwork2)
            
        return actor_loss.item()
    
    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1.0 - self.args.tau) * target_param.data)
            
if __name__ == '__main__':
    args = Args()
    env = gym.make('Pendulum-v1')
    ddpg = TD3Trainer(env=env, args=args, device=torch.device('cpu'))
    ddpg.learn(10000000)
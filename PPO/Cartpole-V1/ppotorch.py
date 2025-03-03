import torch
from torch import nn
import gymnasium as gym
from dataclasses import dataclass
from torch.distributions.categorical import Categorical

@dataclass
class Args:
    gamma: float = 0.99
    lambda_gae: float = 0.95
    epsilon_clip: float = 0.2
    T: int = 2048
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-5
    
@dataclass
class MetricsContainer:
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    entropy_losses = []
    
    def reset(self):
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []

@dataclass
class ExperienceBuffer:
    def __init__(self, T):
        self.T = T
        self.actual_step = 0
        self.states = torch.zeros(T, env.observation_space.shape[0])
        self.actions = torch.zeros(T, 1)
        self.rewards = torch.zeros(T, 1)
        self.next_states = torch.zeros(T, env.observation_space.shape[0])
        self.logprobs = torch.zeros(T, 1)
        self.dones = torch.zeros(T, 1)
        self.advantages = torch.zeros(T, 1)
    
    def append(self, state, action, reward, next_state, logprobs, done):
        self.states[self.actual_step] = state
        self.actions[self.actual_step] = action
        self.rewards[self.actual_step] = reward
        self.next_states[self.actual_step] = next_state
        self.logprobs[self.actual_step] = logprobs
        self.dones[self.actual_step] = done
        self.actual_step += 1
        
    def compute_advantages(self):
        pass
        
class PPOTorch(nn.Module):
    def __init__(self, env, args):
        super(PPOTorch, self).__init__()
        
        # Initialization of env
        self.env = env
        
        # Initialization of ppo variable
        self.args = args
        self.actual_step = 0
        
        # Initialization of actor and critic networks and their optimizers
        self.actor = self.create_network()
        self.critic = self.create_network()
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        
        # Variable used in the algorithm
        self.buffer = ExperienceBuffer(args.T)
        # Variable used in metrics
        self.metrics = MetricsContainer()
        
    def create_network(self):
        return nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.env.action_space.n),
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
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, logprobs, _, _ = self.get_action_and_value(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                
                # I have decided to use a class Experience to make the code easier to understand
                self.buffer.append(state, action, reward, next_state, logprobs, done)
                if self.actual_step % self.args.T == 0 and self.actual_step != 0:
                    self.update()

                self.actual_step += 1
    def update(self):
        for _ in range(self.args.epochs):
            self.buffer.compute_advantages()
            self.update_actor_policy()
            self.update_critic()
    
if __name__ == '__main__':
    args = Args()
    env = gym.make('CartPole-v1')
    ppo = PPOTorch(env, args)
    ppo.train(100)
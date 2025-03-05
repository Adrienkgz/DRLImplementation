import torch
from torch import nn
import gymnasium as gym
from dataclasses import dataclass, field
from torch.distributions.categorical import Categorical
import numpy as np
import time
from tabulate import tabulate

@dataclass
class Args:
    seed: int = None
    gamma: float = 0.99 # Have to be tuned for each environment
    lambda_gae: float = 0.9
    epsilon_clip: float = 0.2
    # Batch size : Length of the buffer to store experiences
    batch_size: int = 2048
    num_envs: int = 4
    mini_batch_size: int = 64
    epochs: int = 10
    lr_actor: float = 3e-4 # Can be tuned
    lr_critic: float = 3e-4 # Can be tuned
    ent_coeff: float = 0.01
    norm_adv: bool = False
    max_grad_norm: float = 0.5
    
    def __post_init__(self):
        self.num_steps = self.batch_size // self.num_envs
    
@dataclass
class MetricsContainer:
    num_steps:int
    episode_rewards:list[float] = field(default_factory=list)
    actor_losses:list[float] = field(default_factory=list)
    critic_losses:list[float] = field(default_factory=list)
    entropy_losses:list[float] = field(default_factory=list)
    entropy_mean:list[float] = field(default_factory=list)
    advantages:None = None
    
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
        datas = [
            ["Number of timesteps", f"{self.num_updates * self.num_steps:.0f}"],
            ["Number of updates", f"{self.num_updates * 10:.0f}"],
            #["Mean reward", f"{np.mean(self.episode_rewards):.2f}"],
            #["Mean episode length", f"{self.num_steps / len(self.episode_rewards):.2f}"],
            ["Time elapsed", f"{(time.time() - self.start_time):.2f} s"],
            ["Steps per second", f"{self.num_steps / (time.time() - self.last_iteration_time):.2f}"],
            ["Mean actor loss", f"{np.mean(self.actor_losses):.2f}"],
            ["Mean critic loss", f"{np.mean(self.critic_losses):.2f}"],
            ["Mean entropy", f"{np.mean(self.entropy_mean):.2f}"],
        ]
        
        print(tabulate(datas, headers=["Metrics", "Value"], tablefmt="fancy_grid"))
        self.reset()

@dataclass
class ExperienceBuffer:
    """
    A buffer to store experiences for PPO (Proximal Policy Optimization) training.
    Attributes:
        ppo (object): The PPO object containing the environment and hyperparameters.
        num_steps (int): The number of steps to store in the buffer.
        cursor (int): The current position in the buffer.
        states (torch.Tensor): A tensor to store states.
        actions (torch.Tensor): A tensor to store actions.
        rewards (torch.Tensor): A tensor to store rewards.
        next_states (torch.Tensor): A tensor to store next states.
        logprobs (torch.Tensor): A tensor to store log probabilities of actions.
        dones (torch.Tensor): A tensor to store done flags.
        advantages (torch.Tensor): A tensor to store computed advantages.
        returns (torch.Tensor): A tensor to store computed returns.
    Methods:
        append(state, action, reward, next_state, logprobs, done):
            Appends a new experience to the buffer.
        compute_advantages_and_returns():
            Computes advantages and returns using Generalized Advantage Estimation (GAE).
        reset():
            Resets the buffer for the next episode.
    """
    def __init__(self, ppo: object):
        """
        Initializes the PPO environment wrapper.

        Args:
            ppo (PPO): The PPO algorithm instance.

        Attributes:
            ppo (PPO): The PPO algorithm instance.
            num_steps (int): Number of steps to run for each environment per update.
            cursor (int): Pointer to the current step in the buffer.
            states (torch.Tensor): Tensor to store states observed from the environment.
            actions (torch.Tensor): Tensor to store actions taken by the agent.
            rewards (torch.Tensor): Tensor to store rewards received from the environment.
            next_states (torch.Tensor): Tensor to store next states observed from the environment.
            logprobs (torch.Tensor): Tensor to store log probabilities of the actions taken.
            dones (torch.Tensor): Tensor to store done flags indicating episode termination.
            advantages (torch.Tensor): Tensor to store advantage estimates.
            returns (torch.Tensor): Tensor to store return estimates.
        """
        self.ppo = ppo
        self.batch_size, num_steps, num_envs = ppo.args.batch_size, ppo.args.num_steps, ppo.args.num_envs
        self.states = torch.zeros((num_steps, num_envs) + ppo.envs.single_observation_space.shape)
        self.actions = torch.zeros((num_steps, num_envs) + ppo.envs.single_action_space.shape)
        self.rewards = torch.zeros((num_steps, num_envs))
        self.next_states = torch.zeros((num_steps, num_envs) + ppo.envs.single_observation_space.shape)
        self.rewards = torch.zeros((num_steps, num_envs))
        self.logprobs = torch.zeros((num_steps, num_envs))
        self.dones = torch.zeros((num_steps, num_envs))
        self.values = torch.zeros((num_steps, num_envs))
        
        self.advantages = torch.zeros(self.batch_size,)
        self.returns = torch.zeros(self.batch_size,)
    
        self.cursor = 0
    
    def append(self, vec_states: np.array, 
               vec_actions: np.array, 
               vec_rewards: np.array, 
               vec_next_states: np.array, 
               vec_logprobs: np.array, 
               vec_dones: np.array,
               vec_values: np.array):
        """
        Append a new experience to the buffer.

        Args:
            state (np.ndarray): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (np.ndarray): The next state of the environment after taking the action.
            logprobs (float): The log probability of the action taken.
            done (bool): A flag indicating whether the episode has ended.

        Returns:
            None
        """
        self.states[self.cursor] = torch.tensor(vec_states)
        self.actions[self.cursor] = vec_actions
        self.rewards[self.cursor] = torch.tensor(vec_rewards)
        self.next_states[self.cursor] = torch.tensor(vec_next_states)
        self.logprobs[self.cursor] = torch.tensor(vec_logprobs)
        self.dones[self.cursor] = torch.tensor(vec_dones)
        self.values[self.cursor] = vec_values.view(-1)
        self.cursor += 1
        
    def flatten(self):
        self.flatten_states = self.states.flatten(0, 1)
        self.flatten_actions = self.actions.flatten()
        self.flatten_rewards = self.rewards.flatten()
        self.flatten_next_states = self.next_states.flatten(0, 1)
        self.flatten_logprobs = self.logprobs.flatten()
        self.flatten_dones = self.dones.flatten()
        self.flatten_values = self.values.flatten()
    
    def compute_advantages_and_returns(self):
        """
        Compute the Generalized Advantage Estimation (GAE) and returns for the current batch of experiences.
        This method calculates the advantages and returns using the Generalized Advantage Estimation (GAE) algorithm.
        It iterates over the states in reverse order to compute the advantages recursively. The returns are then
        calculated by adding the advantages to the state values.
        The computed advantages and returns are stored in the instance variables `self.advantages` and `self.returns`.
        Additionally, the advantages are also stored in the PPO metrics for logging purposes.
        Args:
            None
        Returns:
            None
        """
        with torch.no_grad():
            # First we compute the values of the states and the next state
            values = self.flatten_values
            next_value = self.ppo.get_value(self.flatten_next_states[-1])

            # We initialize the advantages and the last gae lambda
            advantages = torch.zeros_like(self.flatten_rewards)
            lastgaelam = 0  

            # We iterate over the states in reverse order because the GAE is computed recursively
            for t in reversed(range(self.batch_size)):
                if t == self.batch_size - 1: # If it's the last state so the first iteration of the loop
                    nextnonterminal = 1.0 - self.flatten_dones[-1]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.flatten_dones[t + 1] 
                    nextvalues = values[t + 1]

                # Compute td_error
                # δ = r + gamma * V(s_t+1) - V(s_t)
                delta = self.flatten_rewards[t] + self.ppo.args.gamma * nextvalues * nextnonterminal - values[t]
                
                # Compute advantage
                # A(s_t, a_t) = δ + γ * λ * A(s_t+1, a_t+1) if non-terminal then δ
                lastgaelam = delta + self.ppo.args.gamma * self.ppo.args.lambda_gae * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam

            # Compute the returns
            # R_t = A_t + V(s_t)
            self.returns = advantages + values
            self.advantages = advantages
            
            # Metrics
            self.ppo.metrics.advantages = advantages
            
    def reset(self):
        self.cursor = 0
        
        
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes the weights and biases of a given neural network layer.
    We used an orthogonal Initialization of Weights and Constant Initialization of biases
    Recommended in the 37 implementation details of PPO :
     - https://iclr-blog-track.cleanrl.dev/2022/03/25/ppo-implementation-details/

    Args:
        layer (torch.nn.Module): The neural network layer to initialize.
        std (float, optional): The standard deviation for the orthogonal initialization of the weights. Default is sqrt(2).
        bias_const (float, optional): The constant value to initialize the biases. Default is 0.0.

    Returns:
        torch.nn.Module: The initialized neural network layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOTorch(nn.Module):
    def __init__(self, env_or_env_id: str | gym.Env, args: Args):
        super(PPOTorch, self).__init__()
        
        assert isinstance(env_or_env_id, (str, gym.Env)), "env_or_env_id must be a string or a gym.Env instance."
        assert isinstance(args, Args), "args must be an instance of the Args dataclass."

        if isinstance(env_or_env_id, str):
            self.envs = gym.make_vec(env_or_env_id, args.num_envs, vectorization_mode='sync')
        else:
            self.envs = env_or_env_id
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

            
        # Initialization of ppo variable
        self.args = args
        self.actual_step = 1
        
        # Initialization of actor and critic networks and their optimizers
        self.actor = self.create_network('actor')
        self.critic = self.create_network('critic')
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        
        # Variable used in the algorithm
        self.buffer = ExperienceBuffer(self)
        # Variable used in metrics
        self.metrics = MetricsContainer(args.num_steps)
        
        # Initialization seeds ( in case we want to reproduce the results )
        if args.seed is not None:
            import random
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
        
    def create_network(self, net_type):
        """
        Creates a neural network based on the specified network type.
        Args:
            net_type (str): The type of network to create. It can be either 'actor' or 'critic'.
        Returns:
            nn.Sequential: A sequential container of the network layers.
        """
        assert net_type in ['actor', 'critic'], "net_type must be either 'actor' or 'critic'."
        
        # We use the standard architecture for the networks
        layers = [
            layer_init(nn.Linear(self.envs.single_observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        ]
        
        # We add the last layer depending on the network type
        if net_type == 'actor':
            layers.append(layer_init(nn.Linear(64, self.envs.single_action_space.n), std=0.01))
        else:
            layers.append(layer_init(nn.Linear(64, 1), std=1))
        return nn.Sequential(*layers)
        
    def get_action_and_value(self, x: torch.tensor, action: int=None):
        """
        Compute the action to take and its associated value.

        Parameters:
        x (torch.Tensor): The input state tensor.
        action (int, optional): The action to evaluate. If None, a new action is sampled.

        Returns:
        tuple: A tuple containing:
            - action (int): The action to take.
            - log_prob (torch.Tensor): The log probability of the action.
            - entropy (torch.Tensor): The entropy of the action distribution.
            - value (torch.Tensor): The value of the state as estimated by the critic.
        """
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor instance."
        
        # Compute the logits for the actor network
        logits = self.actor(x)
        
        # We apply the softmax function
        probs = Categorical(logits=logits)
        
        # If the action is not given, this implies that we are in the sampling phase
        # So we sample an action
        if action is None:
            # We sample an action according to the probs distribution
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def get_value(self, x: torch.tensor):
        """
        Computes the value of the given input state using the critic network.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The value of the input state as predicted by the critic network.
        """
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor instance."
        
        return self.critic(x)
    
    def train(self, total_timesteps: int):
        """
        Train the PPO agent for a specified number of timesteps.
        Args:
            total_timesteps (int): The total number of timesteps to train the agent.
        The training loop continues until the actual_step reaches total_timesteps. 
        For each episode, the environment is reset, and the agent interacts with the environment 
        by selecting actions and receiving rewards. The experiences are stored in a buffer, 
        and the agent is updated periodically based on the specified number of steps. 
        Metrics such as episode rewards and entropy mean are recorded for analysis.
        """
        assert isinstance(total_timesteps, int), "total_timesteps must be an integer."
        
        number_updates = total_timesteps // self.args.num_steps
        for _ in range(number_updates):
            vec_states, _ = self.envs.reset(seed=self.args.seed)
            vec_dones = np.zeros(self.args.num_envs)
            
            for _ in range(self.args.num_steps):
                with torch.no_grad():
                    vec_actions, vec_logprobs, vec_entropy, vec_values = self.get_action_and_value(torch.tensor(vec_states))
                vec_next_states, vec_rewards, vec_terminated, vec_truncated, _ = self.envs.step(vec_actions.numpy())
                
                vec_dones = np.logical_or(vec_terminated, vec_truncated)
                
                self.buffer.append(vec_states, vec_actions, vec_rewards, vec_next_states, vec_logprobs, vec_dones, vec_values)
                
            self.learn()
            self.metrics.show()
                
    def learn(self):
        """
        Updates the actor and critic networks using the data stored in the buffer.

        This method performs the following steps:
        1. Computes the advantages and returns using the data in the buffer.
        2. Iterates for a number of epochs specified in the arguments:
            a. Updates the actor policy.
            b. Updates the critic.
        3. Resets the buffer after the updates are completed.
        """
        self.buffer.flatten()
        self.buffer.compute_advantages_and_returns()
        for _ in range(self.args.epochs):
            self.update_actor_policy()
            self.update_critic()
        self.buffer.reset()
            
    def update_actor_policy(self):
        """
        Updates the actor policy using the Proximal Policy Optimization (PPO) algorithm.
        This function performs the following steps:
        1. Retrieves old policies, states, advantages, and actions from the buffer.
        2. Shuffles the indices of the steps and creates mini-batches.
        3. For each mini-batch:
            a. Extracts the corresponding states, actions, old policies, and advantages.
            b. Computes new policies and entropy using the current actor network.
            c. Computes the policy loss using the clipped surrogate objective.
            d. Computes the entropy loss and combines it with the policy loss.
            e. Records the loss for metrics.
            f. Performs backpropagation and updates the actor network parameters.
        Args:
            None
        Returns:
            None
        """
        def compute_clip_loss(batch_old_policies: torch.tensor, batch_new_policies: torch.tensor, batch_advantages: torch.tensor):
            """
            Computes the clipped loss for Proximal Policy Optimization (PPO).

            Args:
                batch_old_policies (torch.Tensor): The log probabilities of the actions taken under the old policy.
                batch_new_policies (torch.Tensor): The log probabilities of the actions taken under the new policy.
                batch_advantages (torch.Tensor): The estimated advantages for the actions taken.

            Returns:
                torch.Tensor: The mean clipped loss value.
            """
            # If we want to normalize the advantages
            if self.args.norm_adv:
                batch_advantages = batch_advantages / (torch.abs(batch_advantages).mean() + 1e-8)
                
            # We want to calculate π(a|s) / π_old(a|s)
            # We have log[π(a|s)] and log[π_old(a|s)]
            logratios = batch_new_policies - batch_old_policies
            ratios = torch.exp(logratios)
            
            # We clip the ratios by using a min and max on the ratios policy tensors and clipped the ratio that are
            # more or less than 1 - epsilon_clip or 1 + epsilon_clip
            clipped_ratios = torch.clamp(ratios, 1 - self.args.epsilon_clip, 1 + self.args.epsilon_clip)
            
            # Calculate the loss
            loss = -torch.min(ratios * batch_advantages, clipped_ratios * batch_advantages)
            return loss.mean()
        
        # We retrieve the old policies, states, advantages, and actions from the buffer
        old_policies = self.buffer.flatten_logprobs
        states = self.buffer.flatten_states
        advantages = self.buffer.advantages
        actions = self.buffer.flatten_actions
        
        # We want to shuffle the experiences
        global_indices = torch.randperm(self.args.batch_size)
        for start in range(0, self.args.batch_size, self.args.mini_batch_size):
            end = start + self.args.mini_batch_size
            local_indices = global_indices[start:end]
            batch_states = states[local_indices]
            batch_actions = actions[local_indices]
            batch_old_policies = old_policies[local_indices]
            batch_advantages = advantages[local_indices]
            
            # We compute the new policies and entropy using the current actor network
            _, batch_new_policies, entropy, _ = self.get_action_and_value(batch_states, batch_actions)

            # Compute the loss ( see the function above )
            policy_loss = compute_clip_loss(batch_old_policies, batch_new_policies, batch_advantages)
            
            # We compute the entropy loss
            entropy_loss = -self.args.ent_coeff * torch.mean(entropy)
            
            # Compute the global actor_loss
            actor_loss = policy_loss + entropy_loss
            
            # Metrics
            self.metrics.actor_losses.append(actor_loss.item())
            
            # Backpropagation
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
            self.optimizer_actor.step()
            
    def update_critic(self):
        """
        Update the critic network using the states and returns stored in the buffer.
        This function performs the following steps:
        1. Shuffles the indices of the states and returns.
        2. Iterates over the shuffled indices in batches.
        3. Computes the loss for the critic network using the batch of states and returns.
        4. Appends the loss to the critic_losses metric.
        5. Performs backpropagation and gradient clipping.
        6. Updates the critic network parameters using the optimizer.
        Args:
            None
        Returns:
            None
        """
        def compute_loss_critic(batch_states, returns: torch.tensor):
            """
            Computes the loss for the critic network.

            The loss is calculated as the mean squared error between the predicted 
            values from the critic network and the actual returns.

            Args:
                states (torch.Tensor): The input states to the critic network.
                returns (torch.Tensor): The actual returns to be compared with the 
                            critic network's predictions.

            Returns:
                torch.Tensor: The computed loss value.
            """
            return 0.5 * ((self.critic(batch_states) - returns) ** 2).mean()
        
        # We retrieve the states and returns from the buffer
        states = self.buffer.flatten_states
        returns = self.buffer.returns
        
        # We shuffle the experiences as we done on the actor
        global_indices = torch.randperm(self.args.batch_size)
        for start in range(0, self.args.num_steps, self.args.mini_batch_size):
            end = start + self.args.mini_batch_size
            local_indices = global_indices[start:end]
            batch_states = states[local_indices]
            batch_targets = returns[local_indices]
            
            # Compute the loss
            loss = compute_loss_critic(batch_states, batch_targets)
            
            # Metrics
            self.metrics.critic_losses.append(loss.item())
            
            # Backpropagation
            self.optimizer_critic.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
            self.optimizer_critic.step()
    
if __name__ == '__main__':
    args = Args()
    ppo = PPOTorch(env_or_env_id='CartPole-v1', args=args)
    ppo.train(10000000)
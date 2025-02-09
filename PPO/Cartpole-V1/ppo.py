# I try to implement the PPO algorithm in this file using the tensorflow library.

import tensorflow as tf
import numpy as np
import gymnasium as gym
from collections import deque
from custom_wrapper import create_custom_cartpole
from experience import Experience
from keras.optimizers import Adam

# Parameters
GAMMA = 0.99
LAMBDA_GAE = 0.95
EPSILON_CLIP = 0.2
T = 2048
EPOCHS = 10
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3

class PPOTF:
    def __init__(self, env: gym.Env, batch_size, verbose: int = 0):
        # Initialization of personal variables
        self.verbose = verbose
        self.batch_size = batch_size
        
        # Initialization of algorithm variables
        self.T = T
        self.gamma = GAMMA
        self.lambda_gae = LAMBDA_GAE
        self.epsilon_clip = EPSILON_CLIP
        self.epochs = EPOCHS
        
        # Initialization of functionnal variables for the algorithms
        self.buffer = deque(maxlen=T)
        self.actual_step = 0
        
        # Policy and value networks
        self.actor = self.create_actor()
        self.critic = self.create_critic()
        self.optimizer = Adam()
        
        if self.T % self.batch_size != 0:
            raise ValueError('Batch size must me a multiple of T')
        
    def create_actor(self):
        # Creation of the actor network
        actor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(env.action_space.n, activation='softmax')
        ])
        actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_ACTOR), loss='categorical_crossentropy')
        return actor
    
    def create_critic(self):
        # Creation of the critic network
        critic = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])
        
        critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_CRITIC), loss='mse')
        return critic
    
    def get_action_and_value(self, x, action=None):
        probs = self.actor(x)
        print(probs)
        if action is None:
            action = probs.sample()
        return action, probs[action]
    
    def get_value(self, x):
        return self.critic(x)
        
    def train(self, episodes: int = 1000):
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, logprobs = self.get_action_and_value(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                # I have decided to use a class Experience to make the code easier to understand
                self.buffer.append(Experience(state, action, reward, next_state, logprobs, done))
                
                if self.actual_step % T == 0:
                    self.update()

                self.actual_step += 1
                
    def update(self):
        for _ in range(self.epochs):
            self.compute_advantages()
            self.update_actor_policy()
            self.update_critic()
            
    def compute_advantages(self):   
        # We traverse the buffer in reverse to calculate the advantage
        last_advantage = 0
        for experience in self.buffer.reverse():
            # Compute td_error
            # Formule : δ = r + gamma * V(s_t+1) - V(s_t)
            # => δ = r + gamma * V(s_t+1) - V(s_t) if done = false
            # => δ = r - V(s_t) if done = True
            experience.td_error = experience.reward + self.gamma * self.get_value(experience.next_state) * (1 - experience.done) - self.get_value(experience.state)
            
            #Compute advantage
            if experience.done:
                last_advantage = 0
            
            experience.advantage = experience.td_error + self.gamma * self.lambda_gae * last_advantage
            
    def update_actor_policy(self):
        def compute_clip_loss(batch):
            old_policies = np.array([experience.logprobs for experience in batch])
            _, new_policies = self.get_action_and_value([experience.state for experience in batch])
            advantages = np.array([experience.advantages for experience in batch])
            
            ratio_policy = new_policies / old_policies
            
            unclipped_loss_policy = ratio_policy * advantages
            
            clipped_loss_policy = tf.clip_by_value(ratio_policy, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            
            loss_policy = - tf.minimum(unclipped_loss_policy, clipped_loss_policy).mean()
            
            return loss_policy
            
        number_of_batches = self.T // self.batch_size
        # We update by batch to go faster
        for i in range(0, number_of_batches):
            #Possiblement enlever -1 au in range
            batch = self.buffer[i*self.batch_size:(i+1)*self.batch_size]

            # Update the actor
            loss = compute_clip_loss(batch)
            self.optimizer.minimize(lambda: loss, var_list=self.critic.trainable_variables)

            
    def update_critic(self):
        def compute_loss_critic(batch):
            states = np.array([experience.state for experience in batch])
            
            return 
        
        number_of_batches = self.T // self.batch_size
        # We update by batch to go faster
        for i in range(0, number_of_batches):
            #Possiblement enlever -1 au in range
            batch = self.buffer[i*self.batch_size:(i+1)*self.batch_size]

            # Update the critic
            loss = compute_loss_critic(batch)
            self.optimizer.minimize(lambda: loss, var_list=self.critic.trainable_variables)
            
                
                
            
            
if __name__ == "__main__":
    env = create_custom_cartpole()
    ppo = PPOTF(env, verbose=1)
    ppo.train()
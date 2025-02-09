# I try to implement the PPO algorithm in this file using the tensorflow library.

import tensorflow as tf
import numpy as np
import gymnasium as gym
from collections import deque
from custom_wrapper import create_custom_cartpole
from experience import Experience

# Parameters
GAMMA = 0.99
LAMBDA_GAE = 0.95
EPSILON_CLIP = 0.2
T = 2048
EPOCHS = 10
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3

class PPOTF:
    def __init__(self, env: gym.Env, verbose: int = 0):
        # Initialization of personal variables
        self.verbose = verbose
        
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
    
    def train(self, episodes: int = 1000):
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                # I have decided to use a class Experience to make the code easier to understand
                self.buffer.append(Experience(state, action, reward, next_state, 0))
                
                if self.actual_step % T == 0:
                    self.update()

                self.actual_step += 1
                
    def update(self):
        for _ in range(self.epochs):
            self.compute_advantages()
            
    def compute_advantages(self):
        for experience in self.buffer:
            # Formule : δ = r + lambda (V(s_t+1) - V(s_t))
            
            delta_t = experience.reward + self.lambda_gae * (experience.next_state - experience.state)
            
            experience.advantage =
                
            
                
                
            
            
if __name__ == "__main__":
    env = create_custom_cartpole()
    ppo = PPOTF(env, verbose=1)
    ppo.train()
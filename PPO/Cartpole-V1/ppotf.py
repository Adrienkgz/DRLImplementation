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
LR = 1e-5

class PPOTF:
    def __init__(self, env: gym.Env, batch_size = 32, verbose: int = 0):
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
        self.buffer = []
        self.actual_step = 0
        
        # Policy and value networks
        self.actor = self.create_actor()
        self.critic = self.create_critic()
        self.optimizer = Adam(learning_rate=LR)
        
        if self.T % self.batch_size != 0:
            raise ValueError('Batch size must me a multiple of T')
        
    def create_actor(self):
        # Creation of the actor network
        actor = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(env.observation_space.shape[0],)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(env.action_space.n, activation='softmax')
        ])
        return actor
    
    def create_critic(self):
        # Creation of the critic network
        critic = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(env.observation_space.shape[0],)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])
        return critic
    
    def get_action_and_value(self, x, action=None):
        if isinstance(x, list):
            x = np.array(x)
        if x.ndim == 1:
            x = np.array([x])
        probs = self.actor(x)
        if action is None:
            logits = tf.math.log(probs)
            action = tf.random.categorical(logits, num_samples=1)
            action = tf.squeeze(action).numpy()
        return action, tf.squeeze(probs).numpy()[action]
    
    def get_value(self, x):
        return self.critic(x)
        
    def train(self, episodes: int = 1000):
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state).reshape(1, env.observation_space.shape[0])
            episode_reward = 0
            done = False
            
            while not done:
                action, logprobs = self.get_action_and_value(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.array(next_state).reshape(1, env.observation_space.shape[0])
                
                # I have decided to use a class Experience to make the code easier to understand
                self.buffer.append(Experience(state, action, reward, next_state, logprobs, done))
                if self.actual_step % T == 0 and self.actual_step != 0:
                    self.update()

                self.actual_step += 1
                
    def update(self):
        for _ in range(self.epochs):
            self.compute_advantages()
            self.update_actor_policy()
            self.update_critic()
        self.buffer = []
            
    def compute_advantages(self):   
        # We traverse the buffer in reverse to calculate the advantage
        last_advantage = 0
        inverse_buffer = self.buffer[-1::-1]
        for experience in inverse_buffer:
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
            states = np.array([experience.state for experience in batch]).reshape(len(batch), -1)
            _, new_policies = self.get_action_and_value(states, [int(experience.action) for experience in batch])
            advantages = np.array([experience.advantage for experience in batch])
            
            ratio_policy = new_policies / old_policies
            
            unclipped_loss_policy = ratio_policy * advantages
            
            clipped_loss_policy = tf.clip_by_value(ratio_policy, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            
            loss_policy = - tf.reduce_mean(tf.minimum(unclipped_loss_policy, clipped_loss_policy))
            
            return loss_policy
            
        number_of_batches = self.T // self.batch_size
        # We update by batch to go faster
        for i in range(0, number_of_batches):
            print(self.buffer[0])
            #Possiblement enlever -1 au in range
            batch = self.buffer[i*self.batch_size:(i+1)*self.batch_size]

            states = np.array([experience.state for experience in batch]).reshape(len(batch), -1)
            
            with tf.GradientTape() as tape:
                loss = compute_clip_loss(batch)
            grads = tape.gradient(loss, self.actor.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

            
    def update_critic(self):
        def compute_loss_critic(batch):
            states = np.array([experience.state for experience in batch])
            targets = np.array([experience.reward + self.gamma * self.get_value(experience.next_state) * (1 - experience.done) for experience in batch])
            
            return tf.losses.mean_squared_error(targets, self.critic(states)) 
        
        number_of_batches = self.T // self.batch_size
        # We update by batch to go faster
        for i in range(0, number_of_batches):
            #Possiblement enlever -1 au in range
            batch = self.buffer[i*self.batch_size:(i+1)*self.batch_size]

            states = np.array([experience.state for experience in batch])
            
            with tf.GradientTape() as tape:
                loss = compute_loss_critic(batch)
            grads = tape.gradient(loss, self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
            
                
                
            
            
if __name__ == "__main__":
    env = create_custom_cartpole()
    ppo = PPOTF(env, verbose=1)
    ppo.train()
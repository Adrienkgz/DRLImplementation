# I try to implement the PPO algorithm in this file using the tensorflow library.

import tensorflow as tf
import numpy as np
import gymnasium as gym

# Parameters
GAMMA = 0.99
LAMBDA_GAE = 0.95
EPSILON_CLIP = 0.2
T = 2048
EPOCHS = 10
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3

class PPOTF:
    def __init__(self, verbose: int = 0):
        # Initialization of personal variables
        self.verbose = verbose
        
        # Initialization of algorithm variables
        self.T = 2048
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.epsilon_clip = 0.2
        
    def train(self, episodes: int = 1000):
        for episode in range(episodes):
            
 
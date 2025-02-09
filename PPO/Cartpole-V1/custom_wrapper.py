# This code is used to create a custom reward function for the CartPole-v1 environment.

import gymnasium as gym
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        reward = 0 # Initialize the reward to zero to adjust it
        
        # Bonus reward if the cart is in the center
        reward += 1 - abs(state[0]) / 2.4 # 4.8 is the maximum distance from the center so the bonus reward is between 1 and -1 for the cart position
        
        # Bonus reward if the pole is vertical
        # The pole is vertical when the angle is 0, so the bonus reward is between 1 and -1 for the angle of the pole
        reward += 1 - abs(state[2]) / 0.209
        
        
        # Bonus reward if the action have increased the verticality of the pole
        # We want to help the cartpole to stay vertical, so we give a bonus reward if the pole is more vertical than before
        # But to avoid the cartpole to oscillate we give him a bigger bonus if the pole is already vertical
        reward += abs(state[2]) - abs(state[2]) if abs(state[2]) > 0.1 else 0.5
        return state, reward, done, info

def create_custom_cartpole():
    env = gym.make("CartPole-v1")
    env = CustomRewardWrapper(env)
    return env
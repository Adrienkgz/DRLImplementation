""" 
This file is for the training of the CartpoleV1 environment with the DQN algorithm. And it can also be used to play the game with the weights of the neural network already trained.
The training is done with the following parameters:
- BUFFER_CAPACITY = 1000
- MINIMUM_BUFFER = 100
- BATCH_SIZE = 32
- TARGET_UPDATE_FREQ = 10
- Gamma = 0.95
- Learning rate = 0.001

"""


import gymnasium as gym
import random
import numpy as np
from collections import deque
import os
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

# Définition des paramètres

FILE_PATH_WEIGHT = './CartpoleV1/cartpoleDQNWeights.h5'

BUFFER_CAPACITY = 1000 # Size of the replay buffer
MINIMUM_BUFFER = 100 # Minimum size of the replay buffer to start training
BATCH_SIZE = 32 # Size of the batch to train the neural network
TARGET_UPDATE_FREQ = 10 # Frequency of updating the target neural network
AFFICHE_ENVIRONNEMENT = True # Display the environment

if AFFICHE_ENVIRONNEMENT:
    render = 'human'
else:
    render = 'rgb_array'
TRAINING_COEFFICIENT = 1

# Initialize the environment
env = gym.make('CartPole-v1', render_mode = render )

class CartpoleDQN:
    def __init__(self) -> None:
        self.gamma = 0.95 # Discount factor
        self.epsilon = 1 # Probability of making a random choice
        self.epsilonMin = 0.05 # Minimum probability of making a random choice
        self.epsilonDecay = 0.995 # Decay of the probability of making a random choice
        self.replay_buffer = deque(maxlen=BUFFER_CAPACITY) # Replay buffer
        self.learning_rate = 0.001 # Learning rate of the neural network
        self.model = self.build_model() # Neural network
        self.target_model = self.build_model() # Target neural network
        self.update_target_model() # Update the target neural network

    def do_a_step_in_this_env(self, env, action, state):
        """Performs an action in the environment and returns the new state, the reward and the boolean indicating if the game is over
        This function is used to adjust the reward and the state of the game to make it easier to train the neural network and to have the best comportment of the cartpole
        
        Args:
            env (gym.env): Gymnasium environment
            action (int): Action to take in the environment

        Returns:
            state, reward, done : New state of the environment, reward obtained, boolean indicating if the game is over
        """
        state = state[0]
        new_state, reward, done, _, _ = env.step(action) # Do the action in the environment
        
        reward = 0 # Initialize the reward to zero to adjust it
        
        # Bonus reward if the cart is in the center
        reward += 1 - abs(new_state[0]) / 2.4 # 4.8 is the maximum distance from the center so the bonus reward is between 1 and -1 for the cart position
        
        # Bonus reward if the pole is vertical
        # The pole is vertical when the angle is 0, so the bonus reward is between 1 and -1 for the angle of the pole
        reward += 1 - abs(new_state[2]) / 0.209
        
        
        # Bonus reward if the action have increased the verticality of the pole
        # We want to help the cartpole to stay vertical, so we give a bonus reward if the pole is more vertical than before
        # But to avoid the cartpole to oscillate we give him a bigger bonus if the pole is already vertical
        reward += abs(state[2]) - abs(new_state[2]) if abs(state[2]) > 0.1 else 0.5
        
        new_state = self.formatting_input(new_state) # Format the new state to be in the form of a numpy array (1,4)
        return new_state, reward, done
        
    def build_model(self):
        """ Create a neural network that predicts the q_values of a given state in the form of a (4,1) matrix of the Gymnasium Cartpole environment
        
        Returns:
            model keras : neural network
        """
        # I have chosen an simple architecture because the Cartpole environment is simple
        model = Sequential()
        model.add(Dense(24, activation='relu', input_shape=(4,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Update the target neural network with the weights of the neural network
        """
        self.target_model.set_weights(self.model.get_weights())

    def save_weights_DQN(self):
        """Save the weights of the neural network
        """
        print("Backup in progress, hold on!")
        self.model.save_weights(FILE_PATH_WEIGHT)
        self.dqn_created_bool = True
        print("Backup done")
        return
    
    def load_weights_DQN(self):
        """Load the weights of the neural network
        """
        if os.path.exists(FILE_PATH_WEIGHT):
            print("Loading weights in progress, hold on!")
            self.model.load_weights(FILE_PATH_WEIGHT)
            print("Weights loaded")
        else:
            print("No weights found")
        return

    def formatting_input(self, brut_input):
        """Formats the data so that it is in the form of a numpy array of form (1,4)

        Args:
            brut_input (): Input of the neural network before formatting

        Returns:
            input(np.array): Input to the formatted neural network
        """
        if type(brut_input) != 'numpy.ndarray':
            try:
                brut_input = np.array(brut_input)
            except:
                print("Error in the transformation to array")
        if brut_input.ndim == 1: # Avoid some errors and bugs
            input = brut_input.reshape((1,4))
        else:
            print("Error in the formatting of the matrix: input matrix is of dimension 2 or more ")        
        return input

    def get_action(self, state, without_exploration:bool = False):
        """Takes as input the state of the game and chooses the optimal action, with an epsilon probability of making a random choice to allow the discovery of new possibilities

        Args:
            state (np.array): State of the game in the form of an array (1,4)
            without_exploration (bool, optional): Boolean to know if we want to make a random choice or not. Defaults to False.

        Returns:
            action to take: Either 1 ( go right ) or 0 ( go left )
        """
        if without_exploration:
            e = 0 # If we dont want any exploration, we set the epsilon to 0
        else:
            e = self.epsilon
        if random.uniform(0,1) < e :
            return env.action_space.sample()
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """Adds the transition to its memory buffer

        Args:
            state (np.array): State of the game
            action (int): action taken
            reward (int): award received for action taken
            next_state (np.array): new state of the game after the action taken
            done (bool): Boolean to know if the action has ended the game
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_on_batch(self):
        """Train the neural network from a sample and following the Q-Learning algorithm : q_values = rewards + self.gamma * target_values, with an extra element to know if the action has ended the game that we don't count the q-values of the next state when the game is over
        """

        # We take a sample of the replay buffer
        echantillon = random.sample(self.replay_buffer, BATCH_SIZE)

        # We separate the elements of the sample
        states, actions, rewards, next_states, dones = zip(*echantillon)

        # We format the elements of the sample
        states = np.array(states).reshape((BATCH_SIZE, 4))
        next_states = np.array(next_states).reshape((BATCH_SIZE, 4))
        dones = np.array(dones)

        # We calculate the q_values of the states and the next states
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        target_q_values = self.target_model.predict(next_states, verbose=0)

        # get the max action indice for the next state
        max_action_indices = np.argmax(next_q_values, axis=1)
        # get the target values
        target_values = target_q_values[np.arange(BATCH_SIZE), max_action_indices]
        # update the q_values
        q_values[np.arange(BATCH_SIZE), actions] = rewards + (1 - dones) * self.gamma * target_values
        
        # We train the neural network
        self.model.fit(states, q_values, epochs=1, verbose=1, shuffle=True)


    def play_game(self, without_exploration:bool = False):
        """
        Play a game in the Cartpole environment and train the neural network if necessary
        
        Args:
            without_exploration (bool, optional): Boolean to know if we want to train the neural network. Defaults to False.
            
        Returns:
            reward_total_partie (int): Total reward of the game
            i (int): Number of steps taken to finish the game 
        """
        state = env.reset() # Reset the environment
        state = self.formatting_input(state[0]) # Format the state to be in the form of a numpy array (1,4)
        
        # Initialize the locals variables
        done = False
        reward_total_partie, i = 0, 0
        
        while not done:
            i += 1
            env.render() # Display the environment
            action = self.get_action(state, without_exploration) # Get the action to take
            next_state, reward, done,= self.do_a_step_in_this_env(env, action, state) # Do the action in the environment
            reward_total_partie = reward_total_partie + int(reward) # Update the total reward of the game
            if not without_exploration: # If we are training the neural network
                self.remember(state, action, reward, next_state, done) # Add the transition to the replay buffer
                if len(self.replay_buffer) > MINIMUM_BUFFER: # If the replay buffer is big enough
                    self.train_on_batch() # Train the neural network
            state = next_state # Update the state

        return reward_total_partie, i


    def train_DQN(self, episodes:int = 10, show_game_bool:bool = False):
        """ 
        Train the neural network with the DQN algorithm
        
        Args:
            episodes (int, optional): Number of games to play. Defaults to 10.
            show_game_bool (bool, optional): Boolean to know if we want to display the game. Defaults to False.
            
        Returns:
            bool: True if the training is finished, False otherwise
        """
        for i in range(episodes):
            print(f"Episode {i}") # Display the episode number
            
            reward_total_partie, number_step = self.play_game() # Play a game and train the neural network
            
            # Display the results
            print(f"reward total partie : {reward_total_partie}\nNombre de pas : {number_step}")
            
            if i % TARGET_UPDATE_FREQ == 0 : # If it's time to update the target neural network
                self.update_target_model() # Update the target neural network
                
                # To avoid overfitting, we test the neural network on a game without exploration
                # If the neural network has learned to play the game, we stop the training
                validation_score, number_step_validation = self.play_game(without_exploration=True) # Play a game without exploration
                
                # Display the results
                print(f"\nScore for a game without exploration : {validation_score}\nNombre de pas : {number_step_validation}")
                
                # If the neural network has already learned to play the game, we stop the training
                # I have chosen 950 as a threshold because if the AI is good enough to have a score of 950, it will be able to play infinitely
                if number_step_validation > 950: 
                    print("The model has learned to play Cartpole v1, the training is stopped")
                    self.save_weights_DQN() # Save the weights of the neural network
                    return True
            # Decay the epsilon to make less random choices over time
            if self.epsilon > self.epsilonMin:
                self.epsilon = self.epsilon * self.epsilonDecay
        self.save_weights_DQN() # Save the weights of the neural network after we ended all the episodes
        return True
    
    def enjoy_game(self, number_of_games:int = 10):
        """Play a number of games without training the neural network

        Args:
            number_of_games (int, optional): Number of games to play. Defaults to 10.
        """
        DQN.load_weights_DQN() # Load the weights of the neural network
        scores, steps = [], []
        
        for i in range(number_of_games):
            print(f"Game {i}")
            reward_total_partie, number_step = self.play_game(without_exploration=True)
            print(f"reward total partie : {reward_total_partie}\nNombre de pas : {number_step}")
            scores.append(reward_total_partie)
            steps.append(number_step)
        
        average_score = np.mean(scores)
        average_steps = np.mean(steps)
        std_score = np.std(scores)
        std_steps = np.std(steps)
        
        print(f"Average score: {average_score} +/- {std_score}")
        print(f"Average number of steps: {average_steps} +/- {std_steps}")
        return

DQN = CartpoleDQN()
#If you want to train the neural network change the value of the boolean to True
if False:
    # If you want to train your own neural network from 0, you can comment the next line
    DQN.load_weights_DQN()
    
    # If you want to train the neural network, you can change the number of episodes
    DQN.train_DQN(1000)
else : 
    # If you want to play the game with the weights of the neural network already trained
    DQN.enjoy_game(10)

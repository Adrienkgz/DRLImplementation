import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Définition des paramètres
FILE_PATH_WEIGHT = './CartpoleV1/cartpoleA2CWeights.h5'
GAMMA = 0.95
LEARNING_RATE = 0.001

# Initialize the environment
env = gym.make('CartPole-v1', render_mode='rgb_array')

class CartpoleA2C:
    def __init__(self):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = Adam(learning_rate=LEARNING_RATE)
        self.critic_optimizer = Adam(learning_rate=LEARNING_RATE)

    def build_actor(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def build_critic(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        action_probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_dim, p=action_probs)
        return action

    def do_a_step_in_this_env(self, env, action, state):
        """Performs an action in the environment and returns the new state, the reward and the boolean indicating if the game is over
        This function is used to adjust the reward and the state of the game to make it easier to train the neural network and to have the best comportment of the cartpole
        
        Args:
            env (gym.env): Gymnasium environment
            action (int): Action to take in the environment

        Returns:
            state, reward, done : New state of the environment, reward obtained, boolean indicating if the game is over
        """
        if len(state) == 1:
            state = state[0] # Get the state from the array
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
        return new_state, reward, done
    
    def train(self, episodes=1000):
        for episode in range(episodes):
            state = env.reset()[0]
            episode_reward = 0
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.do_a_step_in_this_env(env, action, state)
                episode_reward += reward

                state = state.reshape(1, self.state_dim)
                next_state = next_state.reshape(1, self.state_dim)

                # Compute TD error
                target = reward + GAMMA * self.critic.predict(next_state) * (1 - int(done))
                td_error = target - self.critic.predict(state)

                # Update critic
                self.critic_optimizer.minimize(lambda: self.critic_loss(state, target), var_list=self.critic.trainable_variables)

                # Update actor
                self.actor_optimizer.minimize(lambda: self.actor_loss(state, action, td_error), var_list=self.actor.trainable_variables)

                state = next_state

            print(f"Episode {episode}: Reward = {episode_reward}")

        self.save_weights()

    def actor_loss(self, state, action, td_error):
        action_probs = self.actor(state)
        action_log_probs = tf.math.log(tf.reduce_sum(action_probs * tf.one_hot(action, self.action_dim), axis=1))
        actor_loss = -tf.reduce_mean(action_log_probs * td_error)
        return actor_loss

    def critic_loss(self, state, target):
        critic_loss = tf.reduce_mean(tf.square(target - self.critic(state)))
        return critic_loss

    def save_weights(self):
        self.actor.save_weights(FILE_PATH_WEIGHT)
        print("Weights saved")

    def load_weights(self):
        if tf.io.gfile.exists(FILE_PATH_WEIGHT):
            self.actor.load_weights(FILE_PATH_WEIGHT)
            print("Weights loaded")
        else:
            print("No weights found")

    def play(self, episodes=10):
        self.load_weights()

        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                env.render()
                action = self.get_action(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward

            print(f"Episode {episode}: Reward = {episode_reward}")

a2c = CartpoleA2C()
a2c.train()
a2c.play()

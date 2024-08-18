import numpy as np
import tensorflow as tf    
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras import backend as K

# from strategy import strategy


class ReplayBuffer:
    def __init__(self, agentIndex, agent_lookback, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.agent_lookback = agent_lookback
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.log_probs = []

    def store_episode(self, states, actions, rewards, dones, next_states, log_probs):
        for state, action, reward, done, next_state, log_prob in zip(states, actions, rewards, dones, next_states, log_probs):
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.next_states.append(next_state)
            self.log_probs.append(log_prob)
        if len(self.states) > self.capacity:
            self.states = self.states[-self.capacity:]
            self.actions = self.actions[-self.capacity:]
            self.rewards = self.rewards[-self.capacity:]
            self.dones = self.dones[-self.capacity:]
            self.next_states = self.next_states[-self.capacity:]
            self.log_probs = self.log_probs[-self.capacity:]            

    # def get_entire_episode(self):
    #     return (
    #         np.array(self.states),
    #         np.array(self.actions),
    #         np.array(self.rewards),
    #         np.array(self.dones),
    #         np.array(self.next_states),
    #         np.array(self.log_probs)
    #     )
    def sample_batch(self):
        idx = np.random.randint(0, len(self.states), size=self.batch_size)
        return (
            np.array(self.states)[idx],
            np.array(self.actions)[idx],
            np.array(self.rewards)[idx],
            np.array(self.dones)[idx],
            np.array(self.next_states)[idx],
            np.array(self.log_probs)[idx]
        )

    def saveEpisodes(self, path):
        data = {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'next_states': self.next_states,
            'log_probs': self.log_probs
        }
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def get_min_max(self):
        data = {
            'states': np.array(self.states).reshape(-1, np.array(self.states).shape[-1]),
            'next_states': np.array(self.next_states).reshape(-1, np.array(self.next_states).shape[-1])
        }
        df = pd.DataFrame(data)
        buff_min = df.min()
        buff_max = df.max()
        buff_mean = df.mean()
        buff_std = df.std()
        return buff_min, buff_max, buff_mean, buff_std

import tensorflow as tf
import tensorflow_probability as tfp

class ActorNetwork(tf.keras.Model):
    def __init__(self, action_dim, hidden_dim, log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.flatten = Flatten()
        self.fc1 = Dense(hidden_dim)
        self.fc2 = Dense(hidden_dim//2)
        self.fc3 = Dense(hidden_dim//4)
        self.mean_layer = Dense(action_dim, activation='sigmoid')
        self.log_std_layer = Dense(action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
    
    def call(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(tf.nn.relu(x))
        x = self.fc3(tf.nn.relu(x))
        mu = self.mean_layer(tf.nn.relu(x))
        log_std = self.log_std_layer(x)
        
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        std = tf.exp(log_std)
        covar = tf.square(std)
        covar = tf.linalg.diag(covar)
        
        return mu, covar

class ValueNetwork(tf.keras.Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = Flatten()
        self.fc1 = Dense(hidden_dim)
        self.fc2 = Dense(hidden_dim//2, activation='relu')
        self.fc3 = Dense(hidden_dim//4, activation='relu')
        self.fc4 = Dense(1)
    
    def call(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(tf.nn.leaky_relu(x))
        return x

class PPOAgent:
    def __init__(self, 
                 agent_lookback, 
                 gamma, 
                 lambda_, 
                 clip_epsilon, 
                 entropy_coef, 
                 critic_coef,  
                 learning_rate,
                 maximize_entropy=True,  # Default to True
                 clip_policy_grads=False,
                 clip_value_grads=False):
        """
        Initialize a Proximal Policy Optimization (PPO) agent.

        Args:
            agent_lookback (int): The number of previous agent states to consider.
            gamma (float): The discount factor.
            lambda_ (float): The parameter for Generalized Advantage Estimation (GAE).
            clip_epsilon (float): The clipping parameter for PPO.
            entropy_coef (float): The coefficient for the entropy term in the actor's loss.
            critic_coef (float): The coefficient for the critic's loss.
            learning_rate (float): The learning rate for the optimizers.
            maximize_entropy (bool, optional): Whether to maximize entropy. Defaults to True.
            clip_policy_grads (bool, optional): Whether to clip policy gradients. Defaults to False.
            clip_value_grads (bool, optional): Whether to clip value gradients. Defaults to False.
        """
        
        # Agent parameters
        self.agent_lookback = agent_lookback
        self.gamma = gamma
        self.maximize_entropy = maximize_entropy
        self.clip_policy_grads = clip_policy_grads
        self.clip_value_grads = clip_value_grads
        self.lambda_ = lambda_
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.learning_rate = learning_rate

        # Network components
        self.actor = None
        self.critic = None

    def build_actor(self, action_dim, hidden_dim, action_std):
        self.actor = ActorNetwork(action_dim, hidden_dim)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Create the covariance vector filled with the specified value
        # self.cov_var = tf.fill([action_dim], action_std * action_std)
        # # # Create the diagonal covariance matrix from the covariance vector
        # self.cov_mat = tf.linalg.diag(self.cov_var)
        
    def build_critic(self, hidden_dim):
        self.critic = ValueNetwork(hidden_dim)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def select_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        # action_mean = self.actor(state)
        # cov_mat = torch.diag(self.action_var)
        mean, covar = self.actor(state)
        # std = tf.exp(log_std)

        value = self.critic(state)
        # std = tf.exp(self.log_std)
        dist = tfd.MultivariateNormalFullCovariance(mean, covar)
        action = dist.sample()
        action = tf.clip_by_value(action, 0, 1)
        return action.numpy()[0], dist.log_prob(action).numpy()[0], value.numpy()[0]

    def discount_reward(self, rewards: np.ndarray, gamma: float=0.99) -> np.ndarray:
        # n_Σ_(k=t) (r_(t + k) * γ ^ (k))
        discounted_rewards = []
        discount_t = 0
        # Start from the last reward and work backward
        for r in reversed(rewards):
            discount_t = discount_t * gamma + r
            discounted_rewards.insert(0, discount_t)
        discounted_rewards = np.array(discounted_rewards)
        return discounted_rewards


    def compute_gaes(self,
            rewards: np.ndarray, 
            values: np.ndarray, 
            gamma: float=0.99, 
            lambda_: float=0.97) -> np.ndarray:
        
        values = np.squeeze(values, axis=1)
        next_values = np.concatenate((values[1:], np.zeros(1)))
        deltas = rewards + (gamma * next_values) - values
        gaes = []
        gae_t = 0
        # Start from the last reward and work backward
        for delta in reversed(deltas):
            gae_t = delta + (gamma * lambda_ * gae_t)
            gaes.insert(0, gae_t)
        gaes = np.array(gaes)
        return gaes  
    
    # @tf.function
    def ppo_update(self, states, actions, old_log_probs, returns, advantages, epochs=20):
        advantages = tf.cast(advantages, dtype=tf.float32)
        old_log_probs = tf.cast(old_log_probs, dtype=tf.float32)
        returns = tf.cast(returns, dtype=tf.float32)    
        actor_l, value_l = [], []
        for i in range(epochs):
            with tf.GradientTape() as tape_policy, tf.GradientTape() as tape_value:
                mean, covar = self.actor(states)
                values = self.critic(states)
                # std = tf.exp(log_std)
                # std = tf.exp(self.log_std)
                dist = tfd.MultivariateNormalFullCovariance(mean, covar)
                new_log_probs = tf.reduce_sum(dist.log_prob(actions), axis=-1, keepdims=True)
                
                ratio = tf.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                actor_loss = -tf.minimum(surr1, surr2)
                if self.maximize_entropy:
                    actor_loss = actor_loss - (self.entropy_coef * dist.entropy())
                else:
                    actor_loss = actor_loss + (self.entropy_coef * dist.entropy())
                actor_loss = tf.reduce_mean(actor_loss)
                actor_l.append(actor_loss)

                value_loss = self.mse_loss(values, returns)
                value_loss = tf.reduce_mean(value_loss)
                value_l.append(value_loss)
            
            grads_actor = tape_policy.gradient(actor_loss, self.actor.trainable_variables)

            if self.clip_policy_grads:
                grads_actor = [tf.clip_by_norm(g, 1.0) for g in grads_actor]
            
            self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))

            grads_critic = tape_value.gradient(value_loss, self.critic.trainable_variables)

            if self.clip_value_grads:
                grads_critic = [tf.clip_by_norm(g, 1.0) for g in grads_critic]

            self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
             
        return np.mean(actor_l), np.mean(value_l)
        
            
    
    def save_policy(self, path):
        self.actor.save(path + '_actor.keras')
        self.critic.save(path + '_critic.keras')
        print("model saved")

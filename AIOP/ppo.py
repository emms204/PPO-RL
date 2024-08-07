import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras import backend as K

tf.debugging.set_log_device_placement(False)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


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

    def build_actor(self, input_dims, log_std_min=-20.0, log_std_max=2.0):
        inputs = Input(shape=(1, self.agent_lookback * input_dims))
        # inputs = Input(shape=(self.agent_lookback, input_dims))
        x = Dense(128, activation='relu')(inputs)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)  
        x = Dense(32, activation='relu')(x)

        mu = Dense(1, activation='sigmoid')(x)  # Changed to sigmoid for [0, 1] bound
        log_std = Dense(1, activation=None)(x)

        log_std_clipped = Lambda(lambda x: tf.clip_by_value(x, log_std_min, log_std_max))(log_std)

        self.actor = Model(inputs, outputs=[mu, log_std_clipped])
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        print(self.actor.summary())

    def build_critic(self, input_dims):
        inputs = Input(shape=(1, self.agent_lookback * input_dims))
        # inputs = Input(shape=(self.agent_lookback, input_dims))
        x = Dense(128, activation='relu')(inputs)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)  # Reduced size to match PyTorch architecture
        x = Dense(32, activation='relu')(x)  # Additional layer
        outputs = Dense(1, activation='linear')(x)  # Output layer with linear activation for value estimation
        
        self.critic = Model(inputs, outputs)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        print(self.critic.summary())

    def select_action(self, state):
        mu, log_std = self.actor.predict(state, verbose=0)
        std = tf.exp(log_std)
        
        # Sample from normal distribution
        normal_dist = tfp.distributions.Normal(mu, std)
        z = normal_dist.sample()
        
        # Squash sample to [0, 1] range
        action = tf.sigmoid(z)
        
        # Compute log probability, accounting for the squashing
        log_prob = normal_dist.log_prob(z) - tf.math.log(action * (1 - action) + 1e-8)
        
        return action, log_prob

    def compute_advantages(self, rewards, dones, values, next_values, normalize=True):
        """
        Compute the advantages and returns for the given rewards, dones, values, and next_values.
        Parameters:
            rewards (numpy.ndarray): An array of shape (n_steps,) representing the rewards at each time step.
            dones (numpy.ndarray): An array of shape (n_steps,) representing whether the episode is done at each time step.
            values (numpy.ndarray): An array of shape (n_steps,) representing the estimated values at each time step.
            next_values (numpy.ndarray): An array of shape (n_steps,) representing the estimated next values at each time step.
            normalize (bool, optional): Whether to normalize the advantages. Defaults to True.
        Returns:
            advantages (numpy.ndarray): An array of shape (n_steps,) representing the computed advantages.
            returns (numpy.ndarray): An array of shape (n_steps,) representing the computed returns.
        """
        # Calculate deltas
        deltas = rewards + self.gamma * (1 - dones) * next_values - values

        # Reverse the deltas and dones to use tf.scan from last to first entry
        reversed_deltas = tf.reverse(deltas, axis=[0])
        reversed_dones = tf.reverse(dones, axis=[0])

        # Define the scan function for computing advantages
        def scan_fn(acc, x):
            delta, done = x
            return delta + self.gamma * self.lambda_ * (1 - done) * acc

        # Initialize the scan with zeros and apply the scan function
        advantages = tf.scan(scan_fn, (reversed_deltas, reversed_dones), initializer=0.0)
        
        # Reverse the advantages to match the original order
        advantages = tf.reverse(advantages, axis=[0])
        returns = advantages + values

        if normalize:
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        return advantages, returns
    
    @tf.function
    def distributed_learn_step(self, states, rewards, dones, next_states, old_log_probs):
        
        values = self.critic(states, training=False)
        values = tf.squeeze(values, axis=[1])
        next_values = self.critic(next_states, training=False)
        next_values = tf.squeeze(next_values, axis=[1])

        advantages, returns = self.compute_advantages(rewards, dones, values, next_values)
        for _ in range(10):
            with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                # Actor update
                mu, log_std = self.actor(states, training=True)
                std = tf.exp(log_std)
                
                # Sample from normal distribution
                normal_dist = tfp.distributions.Normal(mu, std)
                z = normal_dist.sample()
                
                # Squash sample to [0, 1] range
                action = tf.sigmoid(z)
                
                # Compute log probability, accounting for the squashing
                log_prob = normal_dist.log_prob(z) - tf.math.log(action * (1 - action) + 1e-8)

                # Calculate the policy ratio
                policy_ratio = tf.exp(log_prob - old_log_probs)
                
                # Clip the policy ratio to the range [1 - epsilon, 1 + epsilon]
                clipped_policy_ratio = tf.clip_by_value(policy_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                # Compute the policy loss
                actor_loss = -tf.minimum(policy_ratio * advantages, clipped_policy_ratio * advantages)
                if self.maximize_entropy:
                    actor_loss = actor_loss - (self.entropy_coef * normal_dist.entropy())
                else:
                    actor_loss = actor_loss + (self.entropy_coef * normal_dist.entropy())

                actor_loss = tf.reduce_mean(actor_loss)

                # Critic update
                value_pred = self.critic(states, training=True)
                critic_loss = tf.reduce_mean((returns - value_pred) ** 2)

            # Compute gradients and apply updates
            grads_actor = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
            grads_critic = tape_critic.gradient(critic_loss, self.critic.trainable_variables)

            if self.clip_policy_grads:
                grads_actor = [tf.clip_by_norm(g, 1.0) for g in grads_actor]
            if self.clip_value_grads:
                grads_critic = [tf.clip_by_norm(g, 1.0) for g in grads_critic]

            self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

    
    def save_policy(self, path):
        self.actor.save(path + '_actor.keras')
        self.critic.save(path + '_critic.keras')
        print("model saved")

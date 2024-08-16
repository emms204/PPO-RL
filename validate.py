import json
import tensorflow as tf
from functools import partial
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten 
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from AIOP.policy_validate import Policy_Validate
from AIOP.ppo import PPOAgent
import numpy as np
# Data directory
DATA_DIR= './data/norm_data/'
TF_ENABLE_ONEDNN_OPTS=0
# Digital twin experiment files
DT1 = ['FI101_AiPV8969']  # Path to the digital twin file

DT2 = None
DT3 = None
DT4 = None
DT5 = None

# Model directory and name
MODEL_DIR = 'LIC01_AiMVtemp/'
MODEL_NAME = 'LIC01_AiMV_actor.keras'

# log_std_min = -20.0
# log_std_max = 2.0
class ActorNetwork(tf.keras.Model):
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.flatten = Flatten()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.mean_layer = tf.keras.layers.Dense(action_dim)
        self.log_std = tf.Variable(tf.zeros(action_dim), trainable=True)
    
    def call(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        mean = self.mean_layer(x)
        log_std = tf.clip_by_value(self.log_std, -20, 2)
        log_std = tf.broadcast_to(log_std, tf.shape(mean))
        return mean, log_std
    
# custom_clip = partial(tf.clip_by_value, clip_value_min=log_std_min, clip_value_max=log_std_max)


# model = load_model(MODEL_DIR + MODEL_NAME, custom_objects={'Lambda': custom_clip}, safe_mode=False)

# Load the model
# model = load_model(MODEL_DIR + MODEL_NAME, custom_objects={'Lambda': lambda x: tf.clip_by_value(x, log_std_min, log_std_max)}, compile=False)
# custom_objects = {'ppo_loss': PPOAgent.ppo_loss}  # Include any custom loss or metrics functions here

# Pull info from the controller config file
with open(MODEL_DIR + 'config.json', 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

PVindex = config['variables']['pvIndex']
SVindex = config['variables']['svIndex']
MVindex = config['variables']['dependantVar']
agentIndex = config['variables']['independantVars']
agent_lookback = config['agent_lookback']
training_scanrate = config['training_scanrate']
execution_scanrate = config['execution_scanrate']
physics = config['physics']
print(agent_lookback, training_scanrate, execution_scanrate, physics)
EPISODE_LENGTH = 200

actor = ActorNetwork(action_dim=1, hidden_dim=128)  # Recreate the architecture
actor.load_weights(MODEL_DIR + MODEL_NAME)

# Import Validation Tool
val = Policy_Validate(data_dir=DATA_DIR, agentIndex=agentIndex, MVindex=MVindex,
                      SVindex=SVindex, PVindex=PVindex, execution_scanrate=execution_scanrate,
                      training_scanrate=training_scanrate, dt1=DT1, dt2=DT2, dt3=DT3, dt4=DT4, dt5=DT5,
                      episode_length=EPISODE_LENGTH, agent_lookback=agent_lookback, plotIndVars=True)

# Validation Loop
for ep in range(4):
    # Initialize validation loop
    state, done = val.reset()
    # Execute the episode
    for i in range(0, EPISODE_LENGTH):
        # Change controller with max action
        state = state[np.newaxis, :] 
        state = state[::int(training_scanrate / execution_scanrate)].astype('float32')
        # Select the last n samples
        state = state[-agent_lookback:].reshape(1, 1, -1)
        # Run model
        mu, std = actor.predict(state, verbose=0)
        control = mu
        # print("model output: ", control)
        # Advance environment with policy action
        state_, reward, done, info = val.step(control)
        
        # Advance state
        state = state_

        if done: break

    val.plot_eps(MODEL_DIR)

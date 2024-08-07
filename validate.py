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
def build_actor(input_dims, log_std_min=-20.0, log_std_max=2.0):
    inputs = Input(shape=(1, agent_lookback * input_dims))
    # inputs = Input(shape=(self.agent_lookback, input_dims))
    x = Dense(128, activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)  
    x = Dense(32, activation='relu')(x)

    mu = Dense(1, activation='sigmoid')(x)  # Changed to sigmoid for [0, 1] bound
    log_std = Dense(1, activation=None)(x)

    log_std_clipped = Lambda(lambda x: tf.clip_by_value(x, log_std_min, log_std_max))(log_std)

    actor = Model(inputs, outputs=[mu, log_std_clipped])
    return actor
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

actor = build_actor(len(agentIndex)+1)  # Recreate the architecture
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
    print(state[0])
    # Execute the episode
    while not done:
        # Change controller with max action
        state = state[np.newaxis, :] 
        state = state[::int(training_scanrate / execution_scanrate)].astype('float32')
        # Select the last n samples
        state = state[-agent_lookback:].reshape(1, 1, -1)
        # Run model
        mu, covar = actor.predict(state, verbose=0)
        control = mu
        # print("model output: ", control)
        # Advance environment with policy action
        state_, done = val.step(control)
        
        # Advance state
        state = state_

    val.plot_eps(MODEL_DIR)

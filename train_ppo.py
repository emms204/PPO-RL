''' this module executes the PPO algorithm '''
import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import shutil
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
# from silence_tensorflow import silence_tensorflow
from AIOP.sim_maker import Simulator
from AIOP.ppo import ReplayBuffer, PPOAgent
# from AIOP.ppomodified import ReplayBuffer, PPOAgent
from AIOP.reward import RewardFunction
from typing import Tuple, Dict
import time

import IPython
if IPython.get_ipython() is not None:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

# from strategy import strategy



#--------------------Parameters for Simulator-------------------------------
DATA_DIR='data/norm_data/'
DT1 = ['FI101_AiPV8969']
DT2 = None
DT3 = None
DT4 = None
DT5 = None

#what variables do you want the agent to see?
MV_INDEX = 3
PV_INDEX = 1
SV_INDEX = 2

# AGENT_INDEX = [1,2,3,5,6,7]
AGENT_INDEX = [3,5,6,7]
AGENT_LOOKBACK = 5

#about 3-5x as long as the system needs to respond to SP change
EPISODE_LENGTH = 300
#add some noise to the SV to help with simulating responses.
SV_NOISE = 0.05

#--------------------Parameters for Agent-----------------------
CONTROLLER_NAME = 'LIC01_AiMV'
# GAMMA = 0.96

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
# CLIP_EPSILON = 0.18
ENTROPY_COEF = 0.001
# ENTROPY_COEF = 0.0
CRITIC_COEF = 0.5
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
EPOCHS = 10
# EPOCHS = 100

MAX_STEP = 0.05       	#how much can the agent move each timestep
TRAINING_SCANRATE = 5   #scanrate that the dt was trained on and agent trains on.
EXECUTION_SCANRATE = 5  #rate that the model is to be executed

reward_function = RewardFunction(AGENT_INDEX, MV_INDEX, SV_INDEX, PV_INDEX)

######################################################################
#------------Initalize Simulator from trained Environment-------------
######################################################################
sim = Simulator(dt1=DT1, dt2=DT2, dt3=DT3, dt4=DT4, dt5=DT5, data_dir=DATA_DIR, agentIndex=AGENT_INDEX,
                MVindex=MV_INDEX, SVindex=SV_INDEX, agent_lookback=AGENT_LOOKBACK, training_scanrate=TRAINING_SCANRATE,
                episode_length=EPISODE_LENGTH, SVnoise=SV_NOISE
                )

#######################################################################
#--------------Initalize PPO Agent/ReplayBuffer-----------------------
#######################################################################
buff = ReplayBuffer(AGENT_INDEX, agent_lookback=sim.agent_lookback,
                    capacity=1000000, batch_size=BATCH_SIZE)

# Initialize Agent
# with strategy.scope():
agent = PPOAgent(agent_lookback=AGENT_LOOKBACK, gamma=GAMMA, lambda_=LAMBDA, clip_epsilon=CLIP_EPSILON,
                entropy_coef=ENTROPY_COEF, critic_coef=CRITIC_COEF,learning_rate=LEARNING_RATE,
                maximize_entropy=True, clip_policy_grads=True, clip_value_grads=True)
# Build actor and critic networks
agent.build_actor(action_dim=1, hidden_dim=128, action_std=0.5)
agent.build_critic(hidden_dim=128)

################################################################################
#---------------------config file--------------------------------------------
################################################################################

#make a directory to save stuff in
# MODEL_DIR = CONTROLLER_NAME + str(int(round(np.random.rand()*10000, 0))) + '/'
MODEL_DIR = CONTROLLER_NAME + 'temp' + '/'
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
os.mkdir(MODEL_DIR)
print(MODEL_DIR)

#import tag_dictionary
with open('norm_vals.json', 'r', encoding='utf-8') as norm_file:
    tag_dict = json.load(norm_file)

#create a config file
config = {}
config['variables'] = {
    'dependantVar': MV_INDEX,
    'independantVars': AGENT_INDEX,
    'svIndex': SV_INDEX,
    'pvIndex': PV_INDEX,
}
config['agent_lookback'] = AGENT_LOOKBACK
config['training_scanrate'] = TRAINING_SCANRATE
config['execution_scanrate'] = EXECUTION_SCANRATE
config['data_sample_rate'] = sim.timestep
config['gamma'] = GAMMA
config['lambda'] = LAMBDA
config['clip_epsilon'] = CLIP_EPSILON
config['entropy_coef'] = ENTROPY_COEF
config['critic_coef'] = CRITIC_COEF
config['svNoise'] = SV_NOISE
config['episode_length'] = EPISODE_LENGTH
config['max_step'] = MAX_STEP
config['rewards'] = {'general': reward_function.general,
                     'stability': reward_function.stability,
                     'stability_tolerance': reward_function.stability_tolerance,
                     'response': reward_function.response,
                     'response_tolerance': reward_function.response_tolerance
                     }
config['dt'] = {
    'dt1': DT1,
    'dt2': DT2,
    'dt3': DT3,
    'dt4': DT4,
    'dt5': DT5
}

config['physics'] = sim.physics
config['tag_normalize_dict'] = tag_dict

with open(MODEL_DIR + 'config.json', 'w', encoding='utf-8') as outfile:
    json.dump(config, outfile, indent=4)

##############################___PPO___#######################################
# ---------------------For Episode 1, M do------------------------------------
##############################################################################
def rollout(sim: Simulator, agent: PPOAgent) -> Tuple[Dict[str, np.ndarray], float]:

    experience = {
        "states": [],
        "actions": [],
        "rewards": [],
        "values": [],
        "dones": [],
        "log_probs": [],
        "reason": []
    }

    obs, _ = sim.reset()
    eps_reward = 0
    
    for _ in range(0, EPISODE_LENGTH):
        action, log_prob, value = agent.select_action(obs)
        new_obs, reward, done, info = sim.step(action)
        
        experience["states"].append(obs)
        experience["actions"].append(action)
        experience["rewards"].append(reward)
        experience["values"].append(value)
        experience["log_probs"].append(log_prob)
        experience["dones"].append(done)
        experience["reason"].append(info["termination_reason"])

        eps_reward += reward
        obs = new_obs

        if done: break

    experience = {k:np.array(v) for k, v in experience.items()}
    return experience, eps_reward

NUM_EPISODES = 500
normalize_returns = False
normalize_gaes = False

train_performance = {'rewards':[]}
best_rewards = {'episodes': [],'rewards': []}

best_reward = -np.inf
for episode in tqdm(range(0, NUM_EPISODES)):
    # print(f"sim episode_array {sim.episode_array[sim.transition_count]}, sim episodedata {sim.episodedata[sim.transition_count]}")
    data, episode_reward = rollout(sim, agent)
    # for k, v in data.items():
    #    print(k, v.shape)
    data["states"] = (data["states"] - data["states"].mean()) / (data["states"].std() + 1e-5)
    data["returns"] = agent.discount_reward(data["rewards"], GAMMA)
    # data["returns"] = (data["returns"] - data["returns"].mean()) / (data["returns"].std() + 1e-5)
    # data["gaes"] = (data["gaes"] - data["gaes"].mean()) / (data["gaes"]) + 1e-8
    # data["gaes"] = agent.compute_gaes(data["rewards"], data["values"], GAMMA, LAMBDA)
    data["gaes"] = data["returns"] - np.squeeze(data["values"], axis=1)
    data["gaes"] = (data["gaes"] - data["gaes"].mean()) / (data["gaes"] + 1e-5)
    # print(f"Returns: {data["returns"][-10:]}, Gaes: {data["gaes"][-10:]}, Values: {data["values"][-10:]}\n")

    buff.store_episode(data["states"], data["actions"], data["rewards"], 
                       data["dones"], data["states"][1:], data["log_probs"])

    act_loss, crit_loss = agent.ppo_update(data["states"], data["actions"], data["log_probs"], data["returns"], 
                                           data["gaes"], epochs=10)

    train_performance["rewards"].append(episode_reward)

    if episode_reward > best_reward:
        best_reward = episode_reward
        print(f'best total reward for episode {episode+1}: {episode_reward}')
        best_rewards["episodes"].append(episode)
        best_rewards["rewards"].append(best_reward)
        agent.save_policy(MODEL_DIR + CONTROLLER_NAME)

    if (episode + 1) % 10 == 0:
        buff.saveEpisodes(MODEL_DIR + 'replaybuffer.csv')

    print(f'best total reward for episode {episode+1}: {episode_reward:.3f}, Actor Loss: {act_loss:.3f}, Critic Loss: {crit_loss:.3f}')


plt.figure(figsize=(20, 5))
plt.plot(train_performance["rewards"][1:])
plt.scatter(best_rewards["episodes"][1:], best_rewards["rewards"][1:], c="orange")
plt.show()

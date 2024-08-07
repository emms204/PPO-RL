''' this module executes the DQN algorithim following the journal nature paper'''
import os
import json
import logging
import numpy as np
import tensorflow as tf
from AIOP.sim_maker import Simulator
from AIOP.dqn import ReplayBuffer,Agent
from AIOP.reward import RewardFunction


#--------------------Paramaters for Simulator-------------------------------
DATA_DIR='data/norm_data/'
# DT1 = ['5sdt/LI101_AiPV6176',
#        '5sdt/LI101_AiPV9789']
# DT1 = ['FI101_AiPV4503']

DT1 = ['FI101_AiPV8969'] # Okay, Digital Twin
DT2 = None
DT3 = None
DT4 = None
DT5 = None

#what variables do you want the agent to see?
MV_INDEX = 3
PV_INDEX = 1
SV_INDEX = 2

AGENT_INDEX = [1,2,3,5,6,7]
AGENT_LOOKBACK = 5

#about 3-5x as long as the system needs to respond to SP change
# EPISODE_LENGTH = 250#Original One
EPISODE_LENGTH = 200  #Test One
#add some noise to the SV to help with simulating responses.
SV_NOISE = 0.05

#--------------------Paramaters for Agent-----------------------
CONTROLLER_NAME = 'LIC01_AiMV'
GAMMA = 0.96
# EPSILON_DECAY = 0.99995#Original one
EPSILON_DECAY = 0.60  #Test#0.2 is the best in the ablation study
MAX_STEP = 0.05       	#how much can the agent move each timestep
TRAINING_SCANRATE = 5   #scanrate that the dt was trained on and agent trains on.
EXECUTION_SCANRATE = 5  #rate that the model is to be executed


reward_function = RewardFunction(AGENT_INDEX,MV_INDEX,SV_INDEX,PV_INDEX)

######################################################################
#------------Initalize Simulator from trained Environment-------------
######################################################################
sim = Simulator(dt1=DT1,dt2=DT2,dt3=DT3,dt4=DT4,dt5=DT5,data_dir=DATA_DIR,agentIndex=AGENT_INDEX,
                MVindex=MV_INDEX,SVindex=SV_INDEX,agent_lookback=AGENT_LOOKBACK,training_scanrate=TRAINING_SCANRATE,
                episode_length=EPISODE_LENGTH,SVnoise=SV_NOISE
                )

#######################################################################
#--------------Initalize DQN Agent/ReplayBuffer-----------------------
#######################################################################
buff = ReplayBuffer(AGENT_INDEX,agent_lookback=sim.agent_lookback,
                        capacity=1000000,batch_size=64)
                            
#Initialize Agent
agent = Agent(agentIndex=AGENT_INDEX,MVindex=MV_INDEX,agent_lookback=AGENT_LOOKBACK,
            default_agent=False,gamma=GAMMA,epsilon_decay=EPSILON_DECAY,
            min_epsilon=.01,max_step=MAX_STEP,training_scanrate=TRAINING_SCANRATE,
            execution_scanrate=EXECUTION_SCANRATE)

#you have the option to explore with a PID controller, or comment out for random exploration
#agent.PID(100,10,0,PV_INDEX,SV_INDEX,scanrate=5)

agent.buildQ(fc1_dims=128,fc2_dims=128,lr=.002)
agent.buildPolicy(fc1_dims=64,fc2_dims=64,lr=.002)

################################################################################
#---------------------config file--------------------------------------------
################################################################################

#make a directory to save stuff in
MODEL_DIR = CONTROLLER_NAME + str(int(round(np.random.rand()*10000,0))) + '/'
os.mkdir(MODEL_DIR)
print(MODEL_DIR)

#import tag_dictionary
with open('norm_vals.json', 'r', encoding='utf-8') as norm_file:
    tag_dict = json.load(norm_file)

#create a config file
config = {}
config['variables'] = {
                        'dependantVar':MV_INDEX,
                        'independantVars':AGENT_INDEX,
                        'svIndex':SV_INDEX,
                        'pvIndex':PV_INDEX,
                        }
config['agent_lookback'] = AGENT_LOOKBACK
config['training_scanrate'] = TRAINING_SCANRATE
config['execution_scanrate'] = EXECUTION_SCANRATE
config['data_sample_rate'] = sim.timestep
config['epsilon_decay'] = EPSILON_DECAY
config['gamma'] = GAMMA
config['svNoise'] = SV_NOISE
config['episode_length'] = EPISODE_LENGTH
config['max_step'] = MAX_STEP
config['rewards'] = {'general':reward_function.general,
            'stability':reward_function.stability,
            'stability_tolerance':reward_function.stability_tolerance,
            'response': reward_function.response,
            'response_tolerance':reward_function.response_tolerance						
                            }
config['dt'] = {
                'dt1':DT1,
                'dt2':DT2,
                'dt3':DT3,
                'dt4':DT4,
                'dt5':DT5
                }

config['physics'] = sim.physics
config['tag_normalize_dict'] = tag_dict

with open(MODEL_DIR +'config.json', 'w',encoding='utf-8') as outfile:
    json.dump(config, outfile,indent=4)

##############################___DQN___#######################################
# ---------------------For Eposode 1, M do------------------------------------
##############################################################################
#calculate num of episodes to decay epsilon +60.
NUM_EPISODES = int(round(np.log(agent.min_epsilon)/(EPISODE_LENGTH*np.log(EPSILON_DECAY)),0))
print(NUM_EPISODES)
scores = []
print("before training")
# for episode in range(NUM_EPISODES):
for episode in range(EPISODE_LENGTH):

    #reset score
    score = 0
    print("during training")
    #reset simulator
    state,done = sim.reset()

    #initalize the first control position to be the same as the start of the episode data
    control=state[AGENT_LOOKBACK-1,AGENT_INDEX.index(MV_INDEX)]

    #execute the episode
    while not done:

        #select action
        action = agent.selectAction(state)
        
        #change controller with max action
        control += agent.action_dict[action]
        
        #keep controller from going past what the env has seen in training
        control=np.clip(control,sim.MV_min,sim.MV_max)

        # advance Environment with max action and get state_
        state_,done = sim.step(control)
        
        reward = reward_function.calculate_reward(state_, action)

        #Store Transition
        buff.storeTransition(state,action,reward,state_)
        
        #Sample Mini Batch of Transitions
        sample_s,sample_a,sample_r,sample_s_= buff.sampleMiniBatch()

        #fit Q
        agent.qlearn(sample_s,sample_a,sample_r,sample_s_)

        #fit policy
        agent.policyLearn(sample_s)

        # advance state
        state = state_

        # get the score for the episode
        score += reward

    #save a history of episode scores
    scores = np.append(scores,score)
    
    if episode > 25:
        moving_average = np.mean(scores[episode-25:episode])
    else: 
        moving_average = 0
    
    if episode % 10 == 0:
        buff.saveEpisodes(MODEL_DIR + 'replaybuffer.csv') 
        agent.savePolicy(MODEL_DIR + CONTROLLER_NAME)
        
        #Update config with max and min
        with open(MODEL_DIR +'config.json', 'r',encoding='utf-8') as config_file:
            config = json.load(config_file)

        buff_min,buff_max,buff_mean,buff_std= buff.get_min_max()

        min_training_range = {}
        min_95 = {}
        for i in buff_min.index:
            min_training_range[i] = str(buff_min[i])
            min_95[i] = str(buff_mean[i] - 2*buff_std[i])
        config['min_training_range'] = min_training_range
        config['min_95th_pct'] = min_95

        max_training_range = {}
        max_95 = {}
        for i in buff_max.index:
            max_training_range[i] = str(buff_max[i])
            max_95[i] = str(buff_mean[i] + 2*buff_std[i])
        config['max_training_range'] = max_training_range
        config['max_95th_pct'] = max_95

        with open(MODEL_DIR +'config.json', 'w',encoding='utf-8') as outfile:
            json.dump(config, outfile,indent=4)

    print('episode_',episode ,' of ',NUM_EPISODES,' score_', round(score,0),
     ' average_', round(moving_average,0), ' epsilon_', round(agent.epsilon,3))
    
     


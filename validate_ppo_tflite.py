'''validates a dqn model'''
'''validates a ppo model'''

import json
import os
import tensorflow as tf
from AIOP.policy_validate import Policy_Validate

# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)

# silence tf warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#get a list of the training files
DATA_DIR= './data/norm_data/'


#list digital twin experiment files
# DT1 = [
#     '5sdt/LI101_AiPV6176',
#     ]
# DT1 = ['FI101_AiPV4503']#Path to the digital twin file used at the start of ppo building

# DT1 = ['FI101_AiPV4877']#Path to the digital twin file trained for ppo
DT1 = ['FI101_AiPV8969']#Path to the digital twin file trained for dqn

DT2 = None
DT3 = None
DT4 = None
DT5 = None

#list controller models
# MODEL_DIR = 'LIC01_AiMV1435/'
# MODEL_NAME = 'LIC01_AiMV_policy.tflite'

# MODEL_DIR = 'LIC01_AiMVtemp/'
MODEL_DIR = 'Some_Previous_Results_of_PPO_tflite/'
# MODEL_NAME = 'LIC01_AiMV_actor.tflite'
MODEL_NAME = 'LIC01_AiMV_policy.tflite'


# MODEL_DIR = 'LIC01_AiMV8173/'
# MODEL_NAME = 'LIC01_AiMV_actor.tflite'

# Load the TFLite model and allocate tensors.
model = tf.lite.Interpreter(MODEL_DIR+MODEL_NAME)
# model = tf.lite.Interpreter(MODEL_NAME)
model.allocate_tensors()

# Get input and output tensors.
input_details = model.get_input_details()
output_details = model.get_output_details()

#Pull info from the controller config file
with open(MODEL_DIR +'config.json', 'r',encoding='utf-8') as config_file:
    config = json.load(config_file)

PVindex = config['variables']['pvIndex']
SVindex = config['variables']['svIndex']
MVindex = config['variables']['dependantVar']
agentIndex = config['variables']['independantVars']
agent_lookback = config['agent_lookback']
training_scanrate = config['training_scanrate']
execution_scanrate = config['execution_scanrate']
physics = config['physics']

EPISODE_LENGTH = 200

##################################################################
#---------------------Validate Policy----------------------------
##################################################################
#Import Validation Tool
val = Policy_Validate(data_dir=DATA_DIR,agentIndex=agentIndex,MVindex=MVindex,
            SVindex=SVindex,PVindex=PVindex,execution_scanrate=execution_scanrate,
            training_scanrate=training_scanrate,dt1=DT1,dt2=DT2,dt3=DT3,dt4=DT4,dt5=DT5,
            episode_length=EPISODE_LENGTH,agent_lookback=agent_lookback,plotIndVars = True)

#initiate a physics tag for a tank level
# if physics:
# 	input_tags = config['physics_config']['input_tags']
# 	output_tags = config['physics_config']['output_tags']
# 	span_in = config['physics_config']['span_in']
# 	diameter_ft = config['physics_config']['diameter_ft']
# 	orientation = config['physics_config']['orientation']
# 	flow_units = config['physics_config']['flow_units']
# 	length_ft = config['physics_config']['length_ft']
    
# 	val.initiate_physics(pvIndex=PVindex,input_tags=input_tags,output_tags=output_tags,
# 						span_in=span_in,diameter_ft=diameter_ft,orientation=orientation,
# 						flow_units = flow_units,length_ft = length_ft)

for ep in range(4):
# for ep in range(1):

    #initalize validation loop
    state,done = val.reset()
    #execute the episode
    while not done:

        #change controller with max action
        state = state[::int(training_scanrate/execution_scanrate)].astype('float32')
        #select the last n samples
        state = state[-agent_lookback:].reshape(1,agent_lookback,len(agentIndex))
        #run model
        model.set_tensor(input_details[0]['index'], state)
        model.invoke()
        # Changing it
        # control = model.get_tensor(output_details[0]['index'])[0][0]
        control = model.get_tensor(output_details[0]['index'])[0]
        
        #max_control = control.max()
        #print(max_control)
        # argmin_control = control.argmin()
        # print(argmin_control)
        # If we do a change like this, 
        #We shall have a single value for it as an input
        
        #action_value = control
        #action_value ....
        
        # print(control)
        
        #Expected valve position to be controlled?
        #Some how we need to go from action value to action
        # control = 0.5    
        # advance Environment with policy action
        # state_,done = val.step(max_control)

        state_,done = val.step(control)

        # advance state
        state = state_

    val.plot_eps(MODEL_DIR)

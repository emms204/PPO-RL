import logging
import random
import json
import os
# from silence_tensorflow import silence_tensorflow
import numpy as np
import tensorflow as tf
import pandas as pd 
import gym


# silence tf warnings
# silence_tensorflow()
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class RewardScaler:
    def __init__(self):
        self.min_reward = float('inf')
        self.max_reward = float('-inf')

    def scale(self, reward):
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        if self.max_reward > self.min_reward:
            return (reward - self.min_reward) / (self.max_reward - self.min_reward)
        return 0

class Simulator(object):
    def __init__(self,data_dir,agentIndex,MVindex,SVindex,
                agent_lookback=1,training_scanrate=1,dt1=None,dt2=None,dt3=None,dt4=None,dt5=None,
                episode_length=240,SVnoise=0.1):
        """
        Initializes the Simulator object with the provided parameters and sets initial values.
        
        Parameters:
            data_dir (str): The directory containing the data.
            agentIndex (int): Index of the Independent Variables.
            MVindex (int): Index of the Manipulated Variable.
            SVindex (int): Index of the Setpoint Variable.
            agent_lookback (int, optional): Number of time steps to look back for the agent. Defaults to 1.
            training_scanrate (int, optional): The training scan rate. Defaults to 1.
            dt1 (None or data type, optional): Description of dt1. Defaults to None.
            dt2 (None or data type, optional): Description of dt2. Defaults to None.
            dt3 (None or data type, optional): Description of dt3. Defaults to None.
            dt4 (None or data type, optional): Description of dt4. Defaults to None.
            dt5 (None or data type, optional): Description of dt5. Defaults to None.
            episode_length (int, optional): Length of each episode. Defaults to 240.
            SVnoise (float, optional): The noise in the SV. Defaults to 0.1.
        
        Returns:
            None
        """

        with open('timestamps.json', 'r') as savefile:
            self.data_dict = json.load(savefile)

        self.agentIndex = agentIndex
        self.dt1 = dt1
        self.dt2 = dt2
        self.dt3 = dt3
        self.dt4 = dt4
        self.dt5 = dt5
        self.data_dir = data_dir
        self.MVindex = MVindex
        self.SVindex = SVindex
        self.SVnoise = SVnoise
        self.episode_length = episode_length
        self.agent_lookback = agent_lookback
        self.max_lookback = agent_lookback
        self.training_scanrate = training_scanrate
        self.physics = False

        self.reward_scaler = RewardScaler()
        self.general = 1  				#proportional to error signal
        self.stability = 1 				#for stability near setpoint
        self.stability_tolerance =.003	#within this tolerance
        self.response = 1 				#for reaching the setpoint quickly
        self.response_tolerance =.05	#within this tolerance

        self.get_tagnames()
        self.get_min_max()
        
        state_low = np.array([*self.IV_min.values(), self.SV_min])
        state_high = np.array([*self.IV_max.values(), self.SV_max])

        self.state_space = gym.spaces.Box(
            low=np.tile(state_low, self.agent_lookback),
            high=np.tile(state_high, self.agent_lookback),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(low=self.MV_min, high=self.MV_max, shape=(1,), dtype=np.float32)

    def get_tagnames(self):
        """
        Retrieves the tag names associated with the data in the given directory.

        This function reads the CSV file located in the specified data directory and extracts the column names. 
        It then assigns the column names to the `tagnames` attribute of the object.

        The function also prints the name of the Manipulated Variable (MV) and the names of the Independant Variables (IVs).

        Additionally, the function calculates the timestep by subtracting the timestamps of the first two rows of the data 
        and converting the result to seconds. The calculated timestep is assigned to the `timestep` attribute of the object.

        Parameters:
            self (object): The instance of the class.
        
        Returns:
            None
        """
        data = pd.read_csv(self.data_dir + self.data_dict['1']['file'])
        self.tagnames = list(data.columns)

        print(self.tagnames[self.MVindex] + ' is the Manipulated Variable')		
        for indv in self.agentIndex:
            print(self.tagnames[indv] + ' is an Independant Variable')	

        #get timestep
        data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
        self.timestep = pd.Timedelta(data.loc[1,'TimeStamp'] - data.loc[0,'TimeStamp']).seconds

    def get_min_max(self):
        """
        Finds the maximum and minimum values of the SV (setpoint variable) and MV (manipulated variable)
        in the data dictionary.

        This function iterates over each record in the data dictionary and reads the corresponding
        CSV file.
        
        It finds the maximum and minimum values of the SV and MV, and updates the corresponding
        attributes of the object.

        Parameters:
        - self: The instance of the class.

        Returns:
        - None
        """
        
        #find the max and min IVs and SV and MV
        self.IV_max = {}
        self.IV_min = {}
        self.SV_max = 0
        self.SV_min = 1
        self.MV_max = 0
        self.MV_min = 1
        
        for record in self.data_dict:
            data = pd.read_csv(self.data_dir + self.data_dict[record]['file'])
            data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
                
            #get time ranges
            data = data[data['TimeStamp']>self.data_dict[record]['xmin']]
            data = data[data['TimeStamp']<self.data_dict[record]['xmax']]

            data['TimeStamp'] = 0
            data = np.asarray(data).astype('float32')
            
            #get IV_max and IV_min
            for indv in self.agentIndex:
                maxIV = data[:,indv].max()
                minIV = data[:,indv].min()
                if indv in self.IV_max:
                    self.IV_max[indv] = max(self.IV_max[indv],maxIV)
                    self.IV_min[indv] = min(self.IV_min[indv],minIV)
                else:
                    self.IV_max[indv] = maxIV
                    self.IV_min[indv] = minIV
                    
            #get SV_max and SV_min
            maxSV = data[:,self.SVindex].max()
            minSV = data[:,self.SVindex].min()
            self.SV_max = max(self.SV_max,maxSV)
            self.SV_min = min(self.SV_min,minSV)

            #get MV_max and MV_min
            maxMV = data[:,self.MVindex].max()
            minMV = data[:,self.MVindex].min()
            self.MV_max = max(self.MV_max,maxMV)
            self.MV_min = min(self.MV_min,minMV)
            
    def get_data(self):
        """
        Get data from a randomly chosen record in the data dictionary.

        This function selects a random record from the data dictionary and reads the corresponding CSV file.
        It then filters the data based on the time range specified in the record.
        Finally, The data is converted to a numpy array of float32 type and returned.

        Returns:
            numpy.ndarray: The filtered and processed data.
        """
        record = random.choice(list(self.data_dict.keys()))
        data = pd.read_csv(self.data_dir + self.data_dict[record]['file']).iloc[::self.training_scanrate,:]
        data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
                
        #get time ranges
        data = data[data['TimeStamp']>self.data_dict[record]['xmin']]
        data = data[data['TimeStamp']<self.data_dict[record]['xmax']]

        data['TimeStamp'] = 0
        data = np.asarray(data).astype('float32')
        return data
    

    def loadEnv(self):
        """
        Loads the environment model for each DT (Deterministic Tank) specified in the object's attributes.
        
        This function loads the TFLite model and allocates tensors for each DT specified in the object's attributes. 
        It also retrieves the input and output tensors for each DT.
        
        For each DT, the function reads the corresponding 'config.json' file and extracts the necessary information such as the lookback, 
        independant variables, dependant variable, velocity, target minimum and maximum, and scan rate. 
        These values are assigned to the corresponding attributes of the object.
        
        The function also updates the `max_lookback` attribute with the maximum lookback value among all the DTs.
        
        Parameters:
            self (object): The instance of the class.
        
        Returns:
            None
        """
        
        if self.dt1 is not None:
            #load environment model
            dt1 = random.choice(self.dt1)

            # Load the TFLite model and allocate tensors.
            self.dt1_model = tf.lite.Interpreter(model_path= dt1 +'/DT.tflite')
            self.dt1_model.allocate_tensors()

            # Get input and output tensors.
            self.dt1_input_details = self.dt1_model.get_input_details()
            self.dt1_output_details = self.dt1_model.get_output_details()


            with open(dt1+'/config.json', 'r') as savefile:
                dt1_config = json.load(savefile)

            #load input index and lookback
            self.dt1_lookback = dt1_config['dt_lookback']
            self.dt1_independantVars = dt1_config['independantVars']
            self.dt1_dependantVar = dt1_config['dependantVar']
            self.dt1_velocity = dt1_config['velocity']
            self.dt1_targetmin = dt1_config['targetmin']
            self.dt1_targetmax = dt1_config['targetmax']
            self.dt1_scanrate = dt1_config['scanrate']

            self.max_lookback = max(self.dt1_lookback, self.max_lookback)


        if self.dt2 is not None:
            #load environment model
            dt2 = random.choice(self.dt2)

            # Load the TFLite model and allocate tensors.
            self.dt2_model = tf.lite.Interpreter(model_path= dt2 +'/DT.tflite')
            self.dt2_model.allocate_tensors()

            # Get input and output tensors.
            self.dt2_input_details = self.dt2_model.get_input_details()
            self.dt2_output_details = self.dt2_model.get_output_details()


            with open(dt2+'/config.json', 'r') as savefile:
                dt2_config = json.load(savefile)

            #load input index and lookback
            self.dt2_lookback = dt2_config['dt_lookback']
            self.dt2_independantVars = dt2_config['independantVars']
            self.dt2_dependantVar = dt2_config['dependantVar']
            self.dt2_velocity = dt2_config['velocity']
            self.dt2_targetmin = dt2_config['targetmin']
            self.dt2_targetmax = dt2_config['targetmax']
            self.dt2_scanrate = dt2_config['scanrate']

            self.max_lookback = max(self.max_lookback, self.dt2_lookback)

        if self.dt3 is not None:
            #load environment model
            dt3 = random.choice(self.dt3)

            # Load the TFLite model and allocate tensors.
            self.dt3_model = tf.lite.Interpreter(model_path= dt3 +'/DT.tflite')
            self.dt3_model.allocate_tensors()

            # Get input and output tensors.
            self.dt3_input_details = self.dt3_model.get_input_details()
            self.dt3_output_details = self.dt3_model.get_output_details()


            with open(dt3+'/config.json', 'r') as savefile:
                dt3_config = json.load(savefile)

            #load input index and lookback
            self.dt3_lookback = dt3_config['dt_lookback']
            self.dt3_independantVars = dt3_config['independantVars']
            self.dt3_dependantVar = dt3_config['dependantVar']
            self.dt3_velocity = dt3_config['velocity']
            self.dt3_targetmin = dt3_config['targetmin']
            self.dt3_targetmax = dt3_config['targetmax']
            self.dt3_scanrate = dt3_config['scanrate']

            self.max_lookback = max(self.max_lookback,self.dt3_lookback)

        if self.dt4 is not None:
            #load environment model
            dt4 = random.choice(self.dt4)

            # Load the TFLite model and allocate tensors.
            self.dt4_model = tf.lite.Interpreter(model_path= dt4 +'/DT.tflite')
            self.dt4_model.allocate_tensors()

            # Get input and output tensors.
            self.dt4_input_details = self.dt4_model.get_input_details()
            self.dt4_output_details = self.dt4_model.get_output_details()


            with open(dt4+'/config.json', 'r') as savefile:
                dt4_config = json.load(savefile)

            #load input index and lookback
            self.dt4_lookback = dt4_config['dt_lookback']
            self.dt4_independantVars = dt4_config['independantVars']
            self.dt4_dependantVar = dt4_config['dependantVar']
            self.dt4_velocity = dt4_config['velocity']
            self.dt4_targetmin = dt4_config['targetmin']
            self.dt4_targetmax = dt4_config['targetmax']
            self.dt4_scanrate = dt4_config['scanrate']

            self.max_lookback = max(self.max_lookback, self.dt4_lookback)

        if self.dt5 is not None:
            #load environment model
            dt5 = random.choice(self.dt5)

            # Load the TFLite model and allocate tensors.
            self.dt5_model = tf.lite.Interpreter(model_path= dt5 +'/DT.tflite')
            self.dt5_model.allocate_tensors()

            # Get input and output tensors.
            self.dt5_input_details = self.dt5_model.get_input_details()
            self.dt5_output_details = self.dt5_model.get_output_details()


            with open(dt5+'/config.json', 'r') as savefile:
                dt5_config = json.load(savefile)

            #load input index and lookback
            self.dt5_lookback = dt5_config['dt_lookback']
            self.dt5_independantVars = dt5_config['independantVars']
            self.dt5_dependantVar = dt5_config['dependantVar']
            self.dt5_velocity = dt5_config['velocity']
            self.dt5_targetmin = dt5_config['targetmin']
            self.dt5_targetmax = dt5_config['targetmax']
            self.dt5_scanrate = dt5_config['scanrate']

            self.max_lookback = max(self.max_lookback, self.dt5_lookback)

    def reset(self):
        """
        Resets the environment for a new episode.

        This function loads the environment, loads data, and generates a new episode.
        It selects a random data range from the loaded data and adds noise to the SV
        to start in an odd place. It then appends and clips the data to ensure it
        doesn't exceed reality. The function returns the start state and a done flag.

        Returns:
            tuple: A tuple containing the start state (numpy.ndarray) and the done flag (bool).
        """
        # Load environment and data
        self.loadEnv()

        # Load data and select random data range
        data_needed = self.episode_length + self.max_lookback
        data = self.get_data()
        while data_needed > data.shape[0]:
            data = self.get_data()
        
        # Select random data range to generate episode
        startline = random.choice(range(0, data.shape[0] - data_needed))
        endline = startline + data_needed
        self.episodedata = data[startline:endline]

        # Add noise to SV to start in an odd place
        noise = np.random.choice([-1, 1]) * self.SVnoise * np.random.rand()
        self.episodedata[:, self.SVindex] += noise

        # Get the first rows as the start state to return
        start_state = self.episodedata[self.max_lookback - self.agent_lookback:self.max_lookback, self.agentIndex + [self.SVindex]]
        # Flatten the state to match the expected 1D shape
        start_state = start_state.flatten()

        # Ensure the state is within the defined bounds
        start_state = np.clip(start_state, self.state_space.low, self.state_space.high)

        # Ensure the data type matches
        start_state = start_state.astype(np.float32)

        # Initialize episode array
        self.episode_array = np.zeros((data_needed,) + self.episodedata.shape[1:], dtype='float32')
        self.episode_array[:self.max_lookback] = self.episodedata[:self.max_lookback]

        # Initialize counters and flags
        self.transition_count = self.max_lookback
        self.done = False

        return start_state, self.done

    def calculate_reward(self, setpoint, current_flow_rate, current_mv, previous_mv):
        error = setpoint - current_flow_rate
        
        # Base reward inversely proportional to error
        base_reward = -abs(error)
        
        # Stability reward
        stability_reward = -abs(current_mv - previous_mv)
        
        # Bonus for being close to setpoint
        setpoint_bonus = 0
        if abs(error) < self.response_tolerance:
            setpoint_bonus = 1
        elif abs(error) < 2 * self.response_tolerance:
            setpoint_bonus = 0.5
        
        # Combine rewards
        total_reward = base_reward + 0.1 * stability_reward + setpoint_bonus
        
        return total_reward

    def step(self,action):
        """
        This function takes an action and updates the state of the simulation.
        Args:
            action (float): The action to be taken in the simulation.

        Returns:
            tuple: A tuple containing the new state and a boolean flag indicating if the simulation is done.
        """

        # Apply action to the episode array
        # Ensure action is a numpy array and in the correct shape
        action = np.array(action).flatten()
        
        # Clip the action to be within the defined bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.episode_array[self.transition_count,self.MVindex] = action

        # Predict the PVs if required
        if self.dt1 is not None:
            #predict the PV
            dt1_inputs = self.episode_array[self.transition_count-self.dt1_lookback:self.transition_count,self.dt1_independantVars]\
                .reshape(1,self.dt1_lookback,len(self.dt1_independantVars))

            self.dt1_model.set_tensor(self.dt1_input_details[0]['index'], dt1_inputs)
            self.dt1_model.invoke()
            pv_ = self.dt1_model.get_tensor(self.dt1_output_details[0]['index'])[0][0]

            if self.dt1_velocity:
                pv = self.episode_array[self.transition_count-1,self.dt1_dependantVar]
                pv_ = pv + pv_
                
            #overwrite the PV
            self.episode_array[self.transition_count,self.dt1_dependantVar] = np.clip(pv_,0,1)

        if self.dt2 is not None:
            #predict the PV
            dt2_inputs = self.episode_array[self.transition_count-self.dt2_lookback:self.transition_count,self.dt2_independantVars]\
                .reshape(1,self.dt2_lookback,len(self.dt2_independantVars))

            self.dt2_model.set_tensor(self.dt2_input_details[0]['index'], dt2_inputs)
            self.dt2_model.invoke()
            pv_ = self.dt2_model.get_tensor(self.dt2_output_details[0]['index'])[0][0]

            if self.dt2_velocity:
                pv = self.episode_array[self.transition_count-1,self.dt2_dependantVar]
                pv_ = pv + pv_
                
            #overwrite the PV
            self.episode_array[self.transition_count,self.dt2_dependantVar] = np.clip(pv_,0,1)

        if self.dt3 is not None:
            #predict the PV
            dt3_inputs = self.episode_array[self.transition_count-self.dt3_lookback:self.transition_count,self.dt3_independantVars]\
                .reshape(1,self.dt3_lookback,len(self.dt3_independantVars))

            self.dt3_model.set_tensor(self.dt3_input_details[0]['index'], dt3_inputs)
            self.dt3_model.invoke()
            pv_ = self.dt3_model.get_tensor(self.dt3_output_details[0]['index'])[0][0]

            if self.dt3_velocity:
                pv = self.episode_array[self.transition_count-1,self.dt3_dependantVar]
                pv_ = pv + pv_
                
            #overwrite the PV
            self.episode_array[self.transition_count,self.dt3_dependantVar] = np.clip(pv_,0,1)

        if self.dt4 is not None:
            #predict the PV
            dt4_inputs = self.episode_array[self.transition_count-self.dt4_lookback:self.transition_count,self.dt4_independantVars]\
                .reshape(1,self.dt4_lookback,len(self.dt4_independantVars))

            self.dt4_model.set_tensor(self.dt4_input_details[0]['index'], dt4_inputs)
            self.dt4_model.invoke()
            pv_ = self.dt4_model.get_tensor(self.dt4_output_details[0]['index'])[0][0]

            if self.dt4_velocity:
                pv = self.episode_array[self.transition_count-1,self.dt4_dependantVar]
                pv_ = pv + pv_
                
            #overwrite the PV
            self.episode_array[self.transition_count,self.dt4_dependantVar] = np.clip(pv_,0,1)

        if self.dt5 is not None:
            #predict the PV
            dt5_inputs = self.episode_array[self.transition_count-self.dt5_lookback:self.transition_count,self.dt5_independantVars]\
                .reshape(1,self.dt5_lookback,len(self.dt5_independantVars))

            self.dt5_model.set_tensor(self.dt5_input_details[0]['index'], dt5_inputs)
            self.dt5_model.invoke()
            pv_ = self.dt5_model.get_tensor(self.dt5_output_details[0]['index'])[0][0]

            if self.dt5_velocity:
                pv = self.episode_array[self.transition_count-1,self.dt5_dependantVar]
                pv_ = pv + pv_
                
            #overwrite the PV
            self.episode_array[self.transition_count,self.dt5_dependantVar] = np.clip(pv_,0,1)
        
        #if a physics based pv is called then calculate it
        if self.physics:
            self.physics_pv()

        #get the new state to return
        state_ = self.episode_array[self.transition_count-self.agent_lookback+1:self.transition_count+1,self.agentIndex + [self.SVindex]]
        state_ = state_.flatten()
        state_ = np.clip(state_, self.state_space.low, self.state_space.high)
        state_ = state_.astype(np.float32)

        # Calculate reward
        setpoint = self.episode_array[self.transition_count, self.SVindex]
        current_flow_rate = self.episode_array[self.transition_count, self.dt1_dependantVar]
        
        current_mv = self.episode_array[self.transition_count, self.MVindex]
        previous_mv = self.episode_array[self.transition_count - 1, self.MVindex]
        raw_reward = self.calculate_reward(setpoint, current_flow_rate, current_mv, previous_mv)
        self.reward = raw_reward
        # self.reward = self.reward_scaler.scale(raw_reward)
        #check if done
        if self.transition_count > self.episode_length + self.max_lookback-2:
            self.done = True

        #adVance counter
        self.transition_count +=1
        
        return state_, self.reward, self.done
    
    def initiate_physics(self,pvIndex,input_tags,output_tags,span_in,diameter_ft,
                orientation,flow_units,length_ft = 0,):
        self.physics = True
        self.pvIndex = pvIndex
        self.input_tags = input_tags
        self.output_tags = output_tags
        self.span_in = span_in
        self.diameter_ft = diameter_ft
        self.orientation = orientation
        self.flow_units = flow_units
        self.length_ft = length_ft
        #import the min and max values file
        with open('norm_vals.json', 'r',encoding='utf-8') as norm_file:
            tag_dict = json.load(norm_file)
        
        #redo the tag dict to drop tag name
        self.tag_dict = {}
        for index in tag_dict:
            tag = list(tag_dict[index].keys())[0]
            self.tag_dict[index] = [tag_dict[index][tag][0],tag_dict[index][tag][1]]
            
        if self.orientation == 'vertical':
            self.area = 3.14159*(self.diameter_ft/2)**2 #ft3
        elif self.orientation =='horozontal':
            self.area = self.diameter_ft*self.length_ft #estimate...

        if self.flow_units == 'bpd':
            self.volume_conversion = 0.178 #bbl_per_ft3
            self.rate_conversion = (60/(self.training_scanrate*self.timestep))*60*24 #per-day to per-scan
        elif self.flow_units =='gpm':
            self.volume_conversion = 7.48 #gallons_per_ft3
            self.rate_conversion = 60/(self.training_scanrate*self.timestep) #per-minute to per-scan

    def physics_pv(self):
            inflows = 0
            outflows = 0
            for input_tag in self.input_tags:
                inflows += self.episode_array[self.transition_count,input_tag]\
                    *(self.tag_dict[str(input_tag)][0]-self.tag_dict[str(input_tag)][1])\
                    +self.tag_dict[str(input_tag)][1]
            
            for output_tag in self.output_tags:
                outflows += self.episode_array[self.transition_count,output_tag]\
                    *(self.tag_dict[str(output_tag)][0]-self.tag_dict[str(output_tag)][1])\
                    +self.tag_dict[str(output_tag)][1]

            delta_rate = inflows - outflows
            delta_ft3 = delta_rate / self.volume_conversion #ft3/min or ft3/day
            delta_ft_perScan = delta_ft3 / self.area /self.rate_conversion #ft
            dldt = delta_ft_perScan/(self.span_in/12)

            #overwrite thermo calc
            self.episode_array[self.transition_count,self.pvIndex] = \
                np.clip(self.episode_array[self.transition_count-1,self.pvIndex] + dldt ,0,1)

import numpy as np
import tensorflow as tf
import pandas as pd 
import random
import matplotlib.pyplot as plt
import json

#cuDnn bug fix
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class Policy_Validate(object):
	def __init__(self,data_dir,agentIndex,MVindex,SVindex,PVindex,
	      agent_lookback,execution_scanrate=1,training_scanrate = 1,
		  dt1=None,dt2=None,dt3=None,dt4=None,dt5=None,
		  episode_length=600,plotIndVars = False):

		with open('val.json', 'r') as savefile:
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
		self.PVindex=PVindex
		self.episode_length = episode_length
		self.max_lookback = agent_lookback*int(training_scanrate/execution_scanrate)
		self.plotIndVars=plotIndVars
		self.execution_scanrate = execution_scanrate
		self.training_scanrate = training_scanrate
		self.physics = False
		self.get_tagnames()


	def get_tagnames(self):
		data = pd.read_csv(self.data_dir + self.data_dict['1']['file'])
		self.tagnames = list(data.columns)
		data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
		self.timestep = pd.Timedelta(data.loc[1,'TimeStamp'] - data.loc[0,'TimeStamp']).seconds

	def loadEnv(self):
		
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
			print("Start with max lookback: ", self.max_lookback)


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


	def get_data(self):
		record = random.choice(list(self.data_dict.keys()))
		data = pd.read_csv(self.data_dir + self.data_dict[record]['file']).iloc[::self.execution_scanrate,:]
		data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
				
		#get time ranges
		data = data[data['TimeStamp']>self.data_dict[record]['xmin']]
		data = data[data['TimeStamp']<self.data_dict[record]['xmax']]

		data['TimeStamp'] = 0
		data = np.asarray(data).astype('float32')
		return data
	
	def reset(self):
		self.loadEnv()
		#load data 
		data_needed = self.episode_length + self.max_lookback
		data = self.get_data()
		while data_needed > data.shape[0]:
			data = self.get_data()

		#get random data to generate an episode
		startline = random.choice(range(0,data.shape[0]-data_needed))
		endline = startline+data_needed
		self.episodedata = data[startline:endline]

		#get the first rows as the start state to return
		start_state = self.episodedata[:self.max_lookback,self.agentIndex]

		#make an empty array to start the episode
		self.episode_array = np.zeros((self.episodedata.shape),dtype='float32')

		#fill it with enough data to make first line
		self.episode_array[0:self.max_lookback] = self.episodedata[0:self.max_lookback]

		#initalize a counter to keep track of the episode
		self.transition_count = self.max_lookback

		#inatilize a done flag to end the episode
		self.done = False

		#create a variable to show where in the agent_state space the controller positioin is
		self.MVpos = self.agentIndex.index(self.MVindex)

		return start_state,self.done

	def step(self,action):

		#copy episode data to the episode array
		self.episode_array[self.transition_count] = self.episodedata[self.transition_count]
		#overwrite the MV with the agents action in the episode data into the future
		self.episode_array[self.transition_count,self.MVindex] = action
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
		state_ = self.episode_array[self.transition_count-self.max_lookback+1:self.transition_count+1,self.agentIndex]
		
		#check if done
		if self.transition_count > self.episode_length + self.max_lookback-2:
			self.done = True

		#adVance counter
		self.transition_count +=1
		
		return state_,self.done
		

	def plot_eps(self,model_dir):
		self.stats()
		#plot validation response
		fig = plt.figure()
		ax = plt.subplot(111)

		#plot validation response
		ax.plot(self.episodedata[:,self.SVindex], label = self.tagnames[self.SVindex])
		ax.plot(self.episode_array[:,self.PVindex], label = self.tagnames[self.PVindex][:-2]+'AiPV')
		ax.plot(self.episodedata[:,self.PVindex], label = self.tagnames[self.PVindex])
		ax.plot(self.episode_array[:,self.MVindex], label = self.tagnames[self.MVindex][:-2]+'AiMV')
		ax.plot(self.episodedata[:,self.MVindex], label = self.tagnames[self.MVindex])
		
		if self.plotIndVars:
			for indv in self.agentIndex:
				if indv not in[self.SVindex,self.PVindex,self.MVindex]:
					ax.plot(self.episodedata[:,indv], label = self.tagnames[indv])

		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')

		plt.xlabel('time' + str(self.timestep) + '(s)')
		plt.title(label = ' error sum '+str(np.round(self.policy_error,2)), loc='center')
		fig.savefig(model_dir + str(int(np.random.rand()*100000))+'.png')
		fig.clf()

	def stats(self):
		#get error sum
		self.policy_error = 0
		self.PID_error = 0
		for i in range(self.max_lookback,self.episode_length-1):
			self.policy_error += abs(self.episode_array[i,self.SVindex]-self.episode_array[i,self.PVindex])
			self.PID_error += abs(self.episodedata[i,self.SVindex]-self.episodedata[i,self.PVindex])

		
		print('policy error ',self.policy_error, 'PID error ', self.PID_error)

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
		with open('min_max.json', 'r') as norm_file:
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
			self.rate_conversion = (60/(self.execution_scanrate*self.timestep))*60*24 #per-day to per-scan
		elif self.flow_units =='gpm':
			self.volume_conversion = 7.48 #gallons_per_ft3
			self.rate_conversion = 60/(self.execution_scanrate*self.timestep) #per-minute to per-scan

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
				np.clip(self.episode_array[self.transition_count-1,self.pvIndex] + dldt ,0,1)()


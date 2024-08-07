''' digital twin class trains and saves a digital twin model then calls the dt_validate class'''
import json
import os
import numpy as np
import tensorflow as tf
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense,GRU,Activation
from keras.optimizers import Adamax
from AIOP.dt_validate import DTValidate
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors
tf.get_logger().setLevel('ERROR')

#cuDnn bug fix
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class DigitalTwin(object):
    '''creates a Digital Twin'''
    def __init__(self,dt_modelname,data_dir,num_val_sets,val_length,
                independantVars,dependantVar,dt_lookback,velocity,scanrate,plotIndVars):

        with open('timestamps.json', 'r',encoding='utf-8') as savefile:
            self.data_dict = json.load(savefile)
        with open('val.json', 'r',encoding='utf-8') as savefile:
            self.valdata_dict = json.load(savefile)
        self.data_dir = data_dir
        self.num_val_sets = num_val_sets
        self.val_length = val_length
        self.dt_modelname = dt_modelname
        self.dependantVar = dependantVar
        self.independantVars = independantVars
        self.dt_lookback = dt_lookback
        self.velocity = velocity
        self.scanrate=scanrate
        self.plotIndVars = plotIndVars

        #make folder to save stuff in
        self.dt_dir = self.dt_modelname + str(int(round(np.random.rand()*10000,0))) +'/'
        
        self.get_tagnames()
        self.TrainPreprocess()
        self.TrainTestSplit()

    def get_tagnames(self):
        '''gets tag names associated with index values'''
        data = pd.read_csv(self.data_dir + self.data_dict['1']['file'])
        self.tagnames = list(data.columns)

        print(self.tagnames[self.dependantVar] + ' is Dependant Variable')		
        for indv in self.independantVars:
            print(self.tagnames[indv] + ' is an Independant Variable')	

        #get timestep
        data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
        self.timestep = pd.Timedelta(data.loc[1,'TimeStamp'] - data.loc[0,'TimeStamp']).seconds

    def TrainPreprocess (self):
        '''puts data into numpy arrays for training'''
        #using python lists because np.append takes forever...
        self.targets = []
        self.variables = []
        total = 0
        for record in self.data_dict:
            data = pd.read_csv(self.data_dir + self.data_dict[record]['file']).iloc[::self.scanrate,:]
            data['TimeStamp'] = pd.to_datetime(data['TimeStamp'])

            #get time ranges
            data = data[data['TimeStamp']>self.data_dict[record]['xmin']]
            data = data[data['TimeStamp']<self.data_dict[record]['xmax']]

            data['TimeStamp'] = 0
            data = np.asarray(data).astype('float32')

            # data = np.asarray(data)
            new_records = data.shape[0]-self.dt_lookback-1
            total += new_records
            print(record + ' new samples '+ str(new_records) + ' total  ' + str(total))

            var_data = data[:,self.independantVars]
            target_data = data[:,[self.dependantVar]].reshape(data.shape[0])

            for t in range(new_records - self.dt_lookback-1):
                self.variables.append(var_data[t:t+self.dt_lookback,:])
                
                if self.velocity:
                    self.targets.append(target_data[t+self.dt_lookback]-target_data[t-1+self.dt_lookback])
                else:
                    self.targets.append(target_data[t+self.dt_lookback])
                
        
        #convert to np array
        self.targets = np.asarray(self.targets)
        self.variables = np.asarray(self.variables)

        self.targetmin = self.targets.min()
        self.targetmax = self.targets.max()

    def TrainTestSplit (self):
        ''' splits data into an 80/20 train test range'''
        #calculate seperate 80/20 train test val splits
        nx = self.variables.shape[0]
        trainset = int(round(.8*nx,0))

        #Shuffle the Data and create train test dataframes
        perm = np.random.permutation(nx)
        self.x_train = self.variables[perm[0:trainset]]
        self.y_train = self.targets[perm[0:trainset]]
        self.x_test  = self.variables[perm[trainset:nx]]
        self.y_test = self.targets[perm[trainset:nx]]

    def trainDt(self,gru1_dims=64,gru2_dims=64,lr=.01,ep=500,batch_size = 1000):
        '''builds dt tensorflow model and fits it'''
        self.dt_model = Sequential([
            GRU(gru1_dims, input_shape=(self.x_train.shape[1],
                self.x_train.shape[2]),return_sequences=True),
            Activation('linear'),
            GRU(gru2_dims),
            Activation('linear'),
            Dense(1),
            Activation('linear')])

        self.dt_model.compile(optimizer=Adamax(learning_rate=lr), loss='mse')
        print(self.dt_model.summary())
        self.dt_model.fit(self.x_train, self.y_train, 
                          batch_size=batch_size, epochs=ep, verbose=1,
                        validation_data = (self.x_test,self.y_test))
        
        self.saveDt()
        print('Model Saved')
        # self.validate()
        

    def saveDt(self):
        ''' saves dt model as a tflite'''
        #create a experiment folder to save model to
        print(self.dt_dir)
        os.makedirs(self.dt_dir)

        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(self.dt_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
              tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
              tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
  
        tflite_model = converter.convert()

        # Save the model.
        with open(self.dt_dir + 'DT.tflite', 'wb') as f:
            f.write(tflite_model)

        #create a config file 
        config = {
                    'dependantVar':self.dependantVar,
                    'independantVars':self.independantVars,
                    'dt_lookback': self.dt_lookback,
                    'velocity':self.velocity,
                    'targetmin':float(self.targetmin),
                    'targetmax':float(self.targetmax),
                    'scanrate': self.scanrate,
                    'data_sample_rate':self.timestep
                     }
        
        #import tag_dictionary
        with open('norm_vals.json', 'r',encoding = 'utf-8') as norm_file:
            tag_dict = json.load(norm_file)
            config['tag_normalize_dict'] = tag_dict
        
        min_training_range = {}
        max_training_range = {}
        for i in self.independantVars:
            var_index = self.independantVars.index(i)
            min_training_range[i] = str(self.variables[:,:,var_index].min())
            max_training_range[i] = str(self.variables[:,:,var_index].max())
        config['min_training_range'] = min_training_range
        config['max_training_range'] = max_training_range

        with open(self.dt_dir +'config.json', 'w',encoding = 'utf-8') as outfile:
            json.dump(config, outfile, indent=4)
        
        #add in the validation json
        with open(self.dt_dir +'timestamps.json', 'w',encoding = 'utf-8') as outfile:
            json.dump(self.data_dict, outfile, indent=4)

    def validate(self):
        '''calls DTValidate'''
                #initalize the validation class
        self.dt_val = DTValidate(dt_dir=self.dt_dir,data_dir=self.data_dir,
            max_len=self.val_length,plotIndVars=self.plotIndVars)
    
        #initalize validation simulator        
        for i in range(self.num_val_sets):
            #initalize validation loop
            done = self.dt_val.reset()
            
            #execute the episode
            while not done:
                done = self.dt_val.step()

            self.dt_val.plot_eps()



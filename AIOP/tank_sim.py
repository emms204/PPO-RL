import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AIOP.ou_noise import OUActionNoise

class Tank(object):
    '''simulator class for tank problem'''
    def __init__(self,episode_length,scan_rate=1,noise_stdv = 0.01):
        self.diameter = 72 #inches
        self.height = 96 #inches
        self.area = 3.14159*(self.diameter/2)**2
        self.gallons_to_in3 = 231
        self.volume = self.area*self.height/self.gallons_to_in3
        self.state_variables = ['LIC101_PV_%','LIC101_SV_%','LIC101_MV_%',
                              'FI101_PV_gpm','FIC11_PV_gpm','FIC12_PV_gpm',
                            'FIC13_PV_gpm']
        self.pv = 'LIC101_PV_%'
        self.mv = 'LIC101_MV_%'
        self.sv = 'LIC101_SV_%'
        self.pv_index = self.state_variables.index(self.pv)
        self.mv_index = self.state_variables.index(self.mv)
        self.sv_index = self.state_variables.index(self.sv)
        self.episode_length = episode_length
        self.scan_rate = scan_rate
        self.episode_counter = 0
        self.done = False
        self.episode_array = np.zeros((self.episode_length+1,len(self.state_variables)),dtype='float32')
        self.fic1_noise = OUActionNoise(mean=np.zeros(1), 
                                        std_deviation=float(noise_stdv) * np.ones(1))
        self.fic2_noise = OUActionNoise(mean=np.zeros(1), 
                                        std_deviation=float(noise_stdv) * np.ones(1))
        self.fic3_noise = OUActionNoise(mean=np.zeros(1), 
                                        std_deviation=float(noise_stdv) * np.ones(1))

    def reset (self):
        '''initalizes a tank simulation'''
        initial_data = np.random.rand(len(self.state_variables))

        #un-scale variables
        pv = (initial_data[0])*96
        sv = 48
        fic1_in = (initial_data[4])*175
        fic2_in = (initial_data[5])*215
        fic3_in = (initial_data[6])*140
        
        #calculate a the initial cv position so that it starts out stable
        flow_in = (fic1_in + fic2_in + fic3_in)
        action_us = np.log(flow_in/35.594)/0.02679
        flow_out = 35.594*(2.718**(0.02679*action_us))
        initial_data[self.pv_index]=pv*100/96
        initial_data[self.sv_index]=sv*100/96
        initial_data[self.mv_index] = action_us
        initial_data[3]=flow_out
        initial_data[4]=fic1_in
        initial_data[5]=fic2_in
        initial_data[6]=fic3_in

        self.episode_array[0] = initial_data
        state = initial_data

        return state,self.done

    def step(self,action):
        '''steps the simulator foward one time step'''
        mv_flow_rate = 35.594*(2.718**(0.02679*action))
        flow_out = mv_flow_rate*(self.scan_rate / 60)
        fic1_in = np.clip(self.episode_array[self.episode_counter,4]+self.fic1_noise(),0,175)
        fic2_in = np.clip(self.episode_array[self.episode_counter,5]+self.fic2_noise(),0,215)
        fic3_in = np.clip(self.episode_array[self.episode_counter,6]+self.fic3_noise(),0,140)
        flow_in = (fic1_in + fic2_in + fic3_in)*(self.scan_rate/60)

        #calculate new level in inches
        volume_diff = (flow_in - flow_out) * self.gallons_to_in3 #in3
        level_diff = volume_diff / self.area #in
        level_ = (self.episode_array[self.episode_counter,self.pv_index] + level_diff *(self.height/100)) 
        
        #increment episode counter
        self.episode_counter+=1

        #copy data down
        self.episode_array[self.episode_counter] = self.episode_array[self.episode_counter-1]

        #fill in new state space
        self.episode_array[self.episode_counter,self.pv_index] = np.clip(level_,0,100)
        self.episode_array[self.episode_counter,self.mv_index] = action
        self.episode_array[self.episode_counter,3] = mv_flow_rate
        self.episode_array[self.episode_counter,4] = fic1_in
        self.episode_array[self.episode_counter,5] = fic2_in
        self.episode_array[self.episode_counter,6] = fic3_in

        #get state_ to return
        state_ = self.episode_array[self.episode_counter]

        if self.episode_counter > self.episode_length -1:
            self.done = True

        return state_,self.done
    
    def save_eps(self,plot):
        '''saves the data to .csv and plots if True'''
        
        if not os.path.exists('data/'):
            os.mkdir('data/')
        if not os.path.exists('data/raw_data/'):
            os.mkdir('data/raw_data/')

        #create timestamps so we can use app.py
        timestamp = [datetime.datetime(2023,3,1)]
        for t in range(1,self.episode_array.shape[0]):
            timestamp.append(timestamp[t-1]+ datetime.timedelta(seconds = self.scan_rate))
        tme = pd.DataFrame(timestamp, columns=['TimeStamp'])

        eps = pd.DataFrame(self.episode_array, columns=self.state_variables)
        data = pd.concat([tme,eps],axis=1)


        if data.shape[0] > 10000:
            for i in range(data.shape[0] // 10000):
                data[i*10000:(i+1)*10000].to_csv(f'data/raw_data/tank{i}.csv',
                                        sep=',',
                                        index=False,
                                        header=True)

        else:
            data.to_csv('data/raw_data/tank.csv',
                         sep=',',
                         index=False,
                         header=True)

        if plot:
            #plot validation response
            plt.plot(eps[self.sv], label = 'SetPoint')
            plt.plot(eps[self.pv], label = 'PV')
            plt.plot(eps[self.mv], label = 'MV')
            plt.legend(loc='lower right', shadow=True, fontsize='x-large')
            plt.xlabel('time (s)')
            plt.show()
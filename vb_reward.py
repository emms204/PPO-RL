class RewardFunction():
    '''Class to calculate reward'''
    def __init__ (self,AGENT_INDEX,MV_INDEX,SV_INDEX,PV_INDEX):

        self.agentIndex = AGENT_INDEX
        self.MVindex = MV_INDEX
        self.SVindex = SV_INDEX
        self.PVindex = PV_INDEX

        #reward paramaters
        self.general = 1  				#proportional to error signal
        self.stability = 1 				#for stability near setpoint
        self.stability_tolerance =.003	#within this tolerance
        self.response = 1 				#for reaching the setpoint quickly
        self.response_tolerance =.05	#within this tolerance
        self.action_penalty = 0.1 		#penalty for large actions

    def calculate_reward(self,state, action):
        '''CALCULATES REWARD GIVEN STATE SPACE'''
        error = abs(action - state[-1,self.agentIndex.index(self.SVindex)])
        self.reward = self.general - error**2  #use MSE instead of absolute error

        if error < self.response_tolerance:
            self.reward += self.response

        if error < .5*self.response_tolerance:
            self.reward += self.response

        if error < .25*self.response_tolerance:
            self.reward += self.response

        if abs(state[-1,self.agentIndex.index(self.MVindex)]\
            -state[-2,self.agentIndex.index(self.MVindex)]) \
                < self.stability_tolerance and error < self.response_tolerance:
            self.reward += self.stability

        if error < self.response_tolerance:
            self.reward += self.stability

        self.reward -= self.action_penalty * abs(action)  #penalize large actions

        print("Error is: ",error, "Reward is:", self.reward)
        return self.reward

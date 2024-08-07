''' this module is a class for the reward function'''
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

    def calculate_reward(self, state, action):
        '''CALCULATES REWARD GIVEN STATE SPACE'''
        error = abs(action - state[self.agentIndex.index(self.SVindex)])
        self.reward = self.general - error * self.general

        for tolerance in [self.response_tolerance, .5 * self.response_tolerance, .25 * self.response_tolerance]:
            if error < tolerance:
                self.reward += self.response

        if abs(state[self.agentIndex.index(self.MVindex)] - state[self.agentIndex.index(self.MVindex) - 1]) < self.stability_tolerance and error < self.response_tolerance:
            self.reward += self.stability

        if error < self.response_tolerance:
            self.reward += self.stability

        return self.reward

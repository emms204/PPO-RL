class PID():
    ''' a pid loop function'''
    def __init__(self,P,I,D,pvindex,svindex,scanrate=1):
        self.P = P
        self.I = I
        self.D = D
        self.pvindex = pvindex
        self.svindex = svindex
        self.scanrate = scanrate
        self.prev_pv = .5
        self.prev_sv = .5
    
    def step (self,state):
        '''
        pv_delta_errorPrevious = tag.pv_delta_error
        pv_delta_error = pvError - tag.pvPreviousError
        tag.pv_delta_error = pv_delta_error

        Kp = (100.0 / tag.P) * ((tag.MSH - tag.MSL) / (tag.SH - tag.SL))

        Ki = pvError * (_ScanTime / tag.I)

        Kd = (tag.D / _ScanTime) * pv_delta_errorPrevious

        MVDelta = Kp * (pv_delta_error + Ki + Kd)

        tag.MV += MVDelta
        '''
        pv = state[self.pvindex]
        sv = state[self.svindex]
        
        prev_error = self.prev_sv-self.prev_pv

        #update prev values
        self.prev_pv = pv
        self.prev_sv = sv

        error = sv-pv
        
        pv_delta_error = error - prev_error

        kp = self.P
        ki = error *(self.scanrate / self.I)
        kd = (self.D / self.scanrate) * prev_error

        mv_delta = kp * (pv_delta_error + ki + kd)
        
        return mv_delta


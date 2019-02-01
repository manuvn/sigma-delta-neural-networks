import numpy as np
import matplotlib.pyplot as plt


from .Cells import Cells
from .Syn import Syn
from .Vsyn import Vsyn

class adex_neuron(Cells):
    def __init__(self,n=(1,1),Ts=1e-6,
                 Kappa = 0.7, 
                 Temp  = 300,
                 ipCmem  = 2e-12,
                 ipItau  = 30e-12,
                 ipIth   = 30e-12,
                 fbCmem  = 2e-12,
                 fbItau  = 3e-12,
                 fbIth   = 90e-10,
                 fbIin   = 40e-9, # used to set max current output for exponential term
                 eCmem   = 2e-12,
                 eItau    = 30e-9,
                 eIth    = 30e-9,
                 Tp      = 1e-6,
                 thresh  = 0.1e-9,
                 mode    = 'sd',
                 refr    = 1e-3,
                 mismatch = 0):
        
        Cells.__init__(self,n,Ts,mismatch)
        self.Kappa  = Kappa
        self.Temp   = Temp # temperature in kelvin
        self.Ut     = 1.38064852e-23 * self.Temp/1.60217662e-19 # kt/q
        self.Tp     = Tp # Width of an neuron spike
        self.thresh = thresh + thresh * self.mismatch

        self.ipsyn  = Vsyn(n,Ts=Ts,
                 Cmem  = ipCmem,
                 Kappa = Kappa, 
                 Temp  = Temp,
                 Itau  = ipItau,
                 Ith   = ipIth,
                 mismatch = 0)
        
        self.fbsyn     = Vsyn(n,Ts=Ts,
                 Cmem  = fbCmem,
                 Kappa = Kappa, 
                 Temp  = Temp,
                 Itau  = fbItau,
                 Ith   = fbIth,
                 mismatch = mismatch)
        
        self.rsyn  = Syn(n,Ts=Ts,
                 Cmem  = fbCmem,
                 Kappa = Kappa, 
                 Temp  = Temp,
                 Itau  = fbItau,
                 Ith   = 3000*fbIth,
                 Iin   = fbIin,
                 Tp    = Tp,
                 mismatch = mismatch)


        self.eCmem  = eCmem + eCmem *self.mismatch
        self.Itau  = eItau + eItau *self.mismatch
        self.Ith   = eIth + eIth *self.mismatch        
        self.gain  = eIth/eItau # DPI gain
        self.tau   = eCmem*self.Ut/(Kappa * eItau)
        self.incf  = self.gain * (1 - np.exp(-Ts/self.tau))
        self.decf  = np.exp(-self.Ts/self.tau)
        self.Imax  = fbIin

        self.Imem  =  np.zeros(n,dtype=np.bool_) # Holds the current eq to membrane pot        
        self.states = np.zeros(n,dtype=np.bool_) # Holds the spiking information of the corresponding neuron

        self.frecon  = self.fbsyn.states # Exists only to make access simpler
        self.irecon  = self.ipsyn.states
        self.mode   = mode
        self.timer  = np.zeros_like(self.states)
        self.refr   = refr/self.Ts # refr is in seconds
        self.sf     = self.thresh

    def reset_mem(self, nsel=None):
        if nsel == None:
            self.Imem = np.zeros(self.ncells)
        else:
            self.Imem[nsel] = 0
            
    def reset_spikes(self, nsel=None):
        if nsel == None:
            self.states = np.zeros(self.ncells)
        else:
            self.states[nsel] = 0
            
            
    def reset(self, nsel=None):
        self.ipsyn.reset(nsel)
        self.fbsyn.reset(nsel) 
        self.rsyn.reset(nsel) 
        self.reset_mem(nsel)
        self.reset_spikes(nsel)
        
    def upd_state(self):
        # Reset input DPI in case spike event in previous time step.
        # self.ipsyn.reset(self.states) # Resetting input DPI is a waste of power.
#        print('F',self.Imem)
       
        # Update input DPI
        i = self.ipsyn.rd_upd(self.ip)
        
        # Update fb DPI
        s = 0.65*self.fbsyn.rd_upd(i)

        # Update error membrane potential
        nleak = self.Imem * self.decf
        ninc  = (i - s) * self.incf
        self.Imem = ninc + nleak
                    
        # Reset Spike register
        self.reset_spikes(self.states)
        

        if self.timer <= 0:
            self.states = (self.Imem > self.thresh)
            self.timer = self.refr
        else:
            self.states = np.zeros(self.states.shape,dtype=np.bool_)
            self.timer = self.timer - 1
   
        # Reset error dpi in case of spike
        self.reset_mem(self.states) 

        _ = self.rsyn.rd_upd(self.states)

        # Update local storage
        self.irecon = self.ipsyn.states
        self.frecon = self.rsyn.states
        return self.states
    
    def compute(self,ip):
        self.rd_upd(ip)
        return self.frecon

    def print_props(self):
        print("------------------\n")
        print("Neuron properties\n")
        print("Delta = ",self.thresh)
        print("Step time = ",self.Ts)
        print("------------------\n")
        print("Input DPI properties\n")
        self.ipsyn.print_props()
        print("------------------\n")
        print("Feedback DPI properties\n")
        self.fbsyn.print_props()
        print("------------------\n")
        print("Error DPI properties\n")
        print('Time constant = ', self.tau)
        print('Gain = ', self.gain)
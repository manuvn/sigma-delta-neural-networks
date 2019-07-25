import numpy as np
import matplotlib.pyplot as plt


from .Cells import Cells
from .Syn import Syn
from .Vsyn import Vsyn

class Neuron(Cells):
    def __init__(self,n=(1,1),Ts=1e-6,
                 Kappa = 0.7, 
                 Temp  = 300,
                 ipCmem  = 2e-12,
                 ipItau  = 30e-12,
                 ipIth   = 30e-12,
                 fbCmem  = 2e-12,
                 fbItau  = 50e-12,
                 fbIth   = 5e-9,
                 fbIin   = 40e-9,
                 eCmem   = 2e-12,
                 eItau   = 1e-15,
                 eIth    = 100e-15,
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
        
        self.fbsyn  = Syn(n,Ts=Ts,
                 Cmem  = fbCmem,
                 Kappa = Kappa, 
                 Temp  = Temp,
                 Itau  = fbItau,
                 Ith   = fbIth,
                 Iin   = fbIin,
                 Tp    = Tp,
                 mismatch = mismatch)

        self.esyn = Vsyn(n,Ts=Ts,
                 Cmem  = eCmem,
                 Kappa = Kappa, 
                 Temp  = Temp,
                 Itau  = eItau,
                 Ith   = eIth,
                 mismatch = mismatch)
        
        self.states = np.zeros(n,dtype=np.bool_) # Holds the spiking information of the corresponding neuron
        self.frecon  = self.fbsyn.states # Exists only to make access simpler
        self.irecon  = self.ipsyn.states
        self.erecon  = self.ipsyn.states
        self.mode   = mode
        self.timer  = 0
        self.refr   = refr/self.Ts # refr is in seconds
        self.spike_count = np.zeros(n)

    def reset(self, nsel=None):
        if nsel == None:
            self.states = np.zeros(self.ncells)
        else:
            self.states[nsel] = 0
        self.ipsyn.reset(nsel)
        self.fbsyn.reset(nsel) 
        self.esyn.reset(nsel)        
        self.spike_count = 0
                
    def reset(self, nsel=None):
        if nsel.any() == None:
            self.states = np.zeros(self.ncells)
        else:
            self.states[nsel] = 0
            
    def upd_state(self):
        # Reset input DPI in case spike event in previous time step.
        # self.ipsyn.reset(self.states) # Resetting input DPI is a waste of power.

        # Reset error DPI in case spike event in previous time step.
        self.esyn.reset(self.states)
        
        # Update input DPI
        i = self.ipsyn.rd_upd(self.ip)
        
        # Update fb DPI
        s = self.fbsyn.rd_upd(self.states)
        
        # Update error DPI
        if self.mode == 'sd':
            val = i -s
            val[val < 0] = 0
            e = self.esyn.rd_upd(val)
        else:
            e = self.esyn.rd_upd(i)
            
        # Reset Spike register
        self.reset(self.states)
        
        # Check for spiking condition
        # condition to implement spike width implemented in fbsyn
        if self.mode == 'sd': # sigma delta modulation
            self.states = (e > self.thresh)
            self.spike_count += (self.fbsyn.oncount <= 0)*self.states

        elif self.mode == 'dm': # delta modulation
            self.states = (i > (s + self.thresh))
        else: # integrate and fire mode
            if self.timer <= 0:
                self.states = (e > self.thresh)
                self.timer = self.refr
            else:
                self.states = np.zeros(self.states.shape,dtype=np.bool_)
                self.timer = self.timer - 1

        # Update local storage
        self.irecon = self.ipsyn.states
        self.frecon = self.fbsyn.states
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
        self.esyn.print_props()
        print("------------------\n")
        print("Number of spikes = {}\n".format(np.sum(self.spike_count)))
        print("------------------\n")

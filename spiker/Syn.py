import numpy as np
import matplotlib.pyplot as plt
from .Cells import Cells
class Syn(Cells):
    """
    Deals with spiking inputs
    """
    def __init__(self,n=(1,1),Ts=1e-6,
                 Cmem  = 2e-12,
                 Kappa = 0.7, 
                 Temp  = 300,
                 Itau  = 50e-12,
                 Ith   = 5e-9,
                 Iin   = 50e-9,
                 Tp    = 2e-6,
                 mismatch = 0):
        
        Cells.__init__(self,n,Ts,mismatch)
        self.Cmem  = Cmem*(1+self.mismatch) # Membrane cap
        self.Kappa = Kappa
        self.Temp  = Temp # temperature in kelvin
        self.Ut    = 1.38064852e-23 * self.Temp/1.60217662e-19 # kt/q
        self.Itau  = Itau*(1+self.mismatch)
        self.Ith   = Ith*(1+self.mismatch)
        self.Iin   = Iin*(1+self.mismatch)
        self.gain  = Ith/Itau # DPI gain
        self.tau   = Cmem*self.Ut/(Kappa * Itau)
        self.decf  = np.exp(-Ts/self.tau)
        self.Tp    = Tp # Width of an output spike from the neuron
        self.incf  = self.gain * (1 - np.exp(-self.Ts/self.tau)) # Factor by which the internal membrane goes up.
        self.oncount = np.zeros(n)
        self.width = int(Tp/Ts)

    def read_ip(self, ip):
        self.ip = ip
        self.oncount[ip>0] = self.width
        return

    def upd_state(self):
        nleak = self.states * self.decf
        inc_cond = (self.oncount > 0)
        ninc = self.Iin * inc_cond * self.incf
        self.states = ninc + nleak
        self.oncount = self.oncount - 1
        return self.states
    
    def compute(self,ip):
        return self.rd_upd(ip)

    def print_props(self):
        print("Step time = ",self.Ts)
        print("Time constant = ",self.tau)
        print("Gain = ",self.gain)
        print("b = ",  self.Iin * self.gain * (1 - np.exp(-self.Tp/self.tau))) 

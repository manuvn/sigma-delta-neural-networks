import numpy as np
import matplotlib.pyplot as plt

from .Cells import Cells

class Vsyn(Cells):
    """
    Deals with analog inputs
    """
    def __init__(self,n=(1,1),Ts=1e-6,
                 Cmem  = 2e-12,
                 Kappa = 0.7, 
                 Temp  = 300,
                 Itau  = 40e-12,
                 Ith   = 40e-12,
                 mismatch = 0):
        
        Cells.__init__(self,n,Ts,mismatch)
        self.Cmem  = Cmem *(1+self.mismatch) # Membrane cap
        self.Kappa = Kappa 
        self.Temp  = Temp # temperature in kelvin
        self.Ut    = 1.38064852e-23 * self.Temp/1.60217662e-19 # kt/q
        self.Itau  = Itau*(1+self.mismatch)
        self.Ith   = Ith*(1+self.mismatch)
        self.gain  = Ith/Itau # DPI gain
        self.tau   = Cmem*self.Ut/(Kappa * Itau)
        self.incf  = self.gain * (1 - np.exp(-Ts/self.tau))
        self.decf  = np.exp(-self.Ts/self.tau)
    def upd_state(self):
        nleak = self.states * self.decf
        ninc = self.ip * self.incf
        self.states = ninc + nleak
        return self.states
    def print_props(self):
        print("Step time = ",self.Ts)
        print("Time constant = ",self.tau)
        print("Gain = ",self.gain)
        return
    def compute(self,ip):
        return self.rd_upd(ip)

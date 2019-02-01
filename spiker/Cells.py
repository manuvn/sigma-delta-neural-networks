import numpy as np
import matplotlib.pyplot as plt
# cell base class

class Cells(object):
    def __init__(self, n=(1,1), Ts=1e-6, mismatch=0):
        self.Ts = Ts
        self.ncells = n
        self.states = np.zeros(n)
        self.ip = np.zeros(n)
        self.mismatch = mismatch*np.random.rand(n)

    def read_ip(self, ip):
        self.ip = ip
        return
    def upd_state(self):
        self.states = self.ip + 0.0
        return
    def rd_upd(self, ip):
        self.read_ip(ip)
        self.upd_state()
        return self.states
    def reset(self, nsel=None):
        if nsel.any() == None:
            self.states = np.zeros(self.ncells)
        else:
            self.states[nsel] = 0

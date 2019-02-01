import numpy as np
import matplotlib.pyplot as plt

class Tseries(object):

    def __init__(self, value= np.array([]), time = np.array([]), Ts=1e-6):
        self.time = time
        self.value = value
        self.index = 0
        self.Ts = Ts
        return
    
    def __getslice__(self,i,j):
        vals = self.value[i:j]
        times = self.time[i:j]
        return Tseries(vals,times)
    
    def __getitem__(self,t):
        index = self.find_index(t)
        return self.value[index]

    def __setitem__(self,t,value):
        index = self.find_index(t)
        self.value[index] = value + 0.0
        return

    def __add__(self,a):
        vals = self.value + self.a
        return Tseries(vals, self.time)
    
    def __len__(self):
        return len(self.time)
    
    def curr_time(self):
        return self.time[self.index]
    
    def curr_val(self):
        return self.value[self.index]
    
    def suffix(sel,val,t):
        self.time = np.concatenate(self.time,t)
        self.value = np.concatenate(self.value,val)
        return
    
    def concat(self,b,tjump=0):
        val = np.concatenate((self.value,b.value))
        time = np.concatenate((self.time, self.time[-1] + tjump + b.time))
        return Tseries(val,time)
    
    def end_time(self):
        return self.time[-1]
    
    
    def find_index(self,t=0):
        index = len(self.time)-1
        while (index > 0) and (self.time[index] > t):
            index = index - 1
        return index

    def time_slice(self,tstart=0,tstop=None):
        if tstop == None:
            tstop = self.end_time()
        istart = self.find_index(tstart)
        istop  = self.find_index(tstop)
        return Tseries(self.value[istart:istop],self.time[istart:istop])

    
    def plot(self,tstart=0,tstop=None,label="Signal",color=None, xlabel=True, ylabel=True):
        if tstop == None:
            tstop = self.end_time()
        istart = self.find_index(tstart)
        istop  = self.find_index(tstop)
        plt.plot(self.time[istart:istop], self.value[istart:istop],label=label, color=color)
        plt.grid(True)
        if xlabel:
            plt.xlabel(r'$Time$')
        if ylabel:
            plt.ylabel(r'$Amplitude$')
        plt.legend()
        return
    
    def set_time(self,t=0):
        # Sets the internal index to the the closest time step less than or equal to t
        self.index = self.find_index(t)
        return

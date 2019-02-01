import numpy as np
import scipy.io as spio

from .Tseries import Tseries

def gen_time(tstart,tstop,Ts=1e-6):
    # Returns Sampling time and time step values
    tvals = np.linspace(tstart,tstop,int((tstop-tstart)/Ts))
    return Tseries(time=tvals,Ts=Ts)

# Waveform generation functions
def gen_sine(tseries,f=100,amp=1e-9):
    t = tseries.time
    sine = Tseries(amp*np.sin(2*np.pi*t*f)+2*amp,t)
    return sine

def gen_ptrain(tseries,f=1000,amp=1e-9):
    t = tseries.time
    Ts = tseries.Ts
    blocklen = int(0.5/(f*Ts))
    block1 = np.ones(blocklen)
    block0 = np.zeros(blocklen)
    block = np.concatenate((block1,block0))
    numblocks = int(np.floor(tseries.time[-1]*f))
    vals = np.zeros_like(t)
    if numblocks > 0:
        vals[0:numblocks*blocklen*2] = np.tile(block,numblocks)
    return Tseries(amp*vals,t)

def gen_dc(tseries,amp=1e-9):
    t = tseries.time
    dc = Tseries(amp*np.ones_like(t),t)
    return dc

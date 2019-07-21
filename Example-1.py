#!/usr/bin/env python
# coding: utf-8

import re
import importlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from spiker import *


# Generate simulation data
def gen_data(nblocks,f1,f2,tseries,amp=1e-9):
    """
    Generates a time series comprising nblocks of data
    Each block is some combination of two sinusoids.
    nblocks = number of blocks
    tseries = Tseries object. Tseries is a datastructure 
        that makes it easy to deal with time-series data
    amp = Scales the entire data block
    """
    time = tseries.time
    Ts = tseries.Ts
    ip = np.array([])
    op = np.array([])
    tblockend = int(len(time)/nblocks)
    timeblock = time[:tblockend]
    for index in range(nblocks):
        if np.random.randint(2):
            block = 2*np.sin(2*np.pi*timeblock*f2)+1*np.sin(2*np.pi*timeblock*f1)+2
            ip = np.concatenate((ip,block))
            op = np.concatenate((op,np.zeros_like(timeblock)))
        else:
            block = 1*np.sin(2*np.pi*timeblock*f2)+2*np.sin(2*np.pi*timeblock*f1)**2+1
            ip = np.concatenate((ip,block))
            op = np.concatenate((op,np.ones_like(timeblock)))
    return amp*ip, amp*op


# Transient simulation parameters for the spiking simulation
Ts = 1e-6 # step time
tstart = 0 # Simulation start time in seconds
tstop = 1 # Simulation stop time in seconds
tseries = gen_time(tstart,tstop,Ts) # obtain the time series input sample
time = tseries.time # the series of time samples


# Neuron constants to setup the neuron model and reserovoir decay
Cmem  = 2e-12 # Membrane cap
Kappa = 0.7
Temp  = 300 # temperature in kelvin
Ut    = 1.38064852e-23 * Temp/1.60217662e-19 # kt/q
Itau  = 50e-12 #50e-12
Ith   = 50e-12 #5e-9
Iin   = 50e-9
gain  = Ith/Itau # DPI gain
tau   = Cmem*Ut/(Kappa * Itau)
print(f'Time constant is {tau} seconds')


# Test data generation
nblocks = 10
f1 = 15
f2 = 4
amp = 10e-9
test_ip, test_op = gen_data(nblocks,f1,f2,tseries,amp)

ip = Tseries(test_ip,time,Ts)
op = Tseries(test_op,time,Ts)

test_ip = np.vstack((amp*np.ones_like(tseries.time),test_ip)) # adding bias


# --------------------------------
# Standard ESN using floating-point, non-spiking neurons
# --------------------------------
def realesn(inSize=1, 
            outSize=1, 
            resSize=20):

    """
    inSize = input dimension
    outSize = output dimension
    resSize = size of the reservoir
    """
    
    # Generate input connectivity weights
    Win = (np.random.rand(resSize,1+inSize)-0.5) * 1

    # Recurrent connecivity weights
    W = np.random.rand(resSize,resSize)-0.5
    # compute the spectral radius
    radius = np.max(np.abs(np.linalg.eigvals(W)))
    # rescale to desired spectral radius
    W = W * (1.0 / radius)

    # Output weights
    Wout = np.random.rand(outSize,1+inSize+resSize)-0.5

    # Generate graph for visualization
    G,W_full = create_graph(W,Win,Wout)
    plt.tight_layout()
    # plt.savefig('ESNgraph.pdf')
    
    nnodes,nip,nnip,nnres,nop = compute_nodes(Win,Wout,W)    
    op = np.zeros((outSize,len(tseries.time)))
    x = np.zeros((resSize,1))
    states = np.zeros((resSize,len(tseries.time)))
    res = np.zeros((outSize,1))
    
    # the retention factor inside reservoir = exp(-nsamples_in_tau/a)
    # chosen so that in time 10*tau, the value of exponent is almost 0
    a = 1 - np.exp(-Ts/tau)
    
    for t in range(len(test_ip[0,:])):
        u = test_ip[:,t]
        u.shape = (len(u),1)
        data_in = np.dot( Win, u)
        data_in.shape = (resSize,1)

        x_ip = data_in + np.dot( W, x )
        x_ip = np.maximum(x_ip,0)
        x = (1-a)*x + a*x_ip
        
        res_ip = np.dot(Wout, np.vstack((u,x)))
        res_ip = np.maximum(res_ip,0)
        res = (1-a)*res + a*res_ip

        # Save state transient output
        x.shape = (resSize,)
        states[:,t] = x + 0
        op[:,t] = res[:,0]
        x.shape = (resSize,1)
    return op, states, Win, W, Wout


# define reservoir properties
inSize, outSize, resSize = 1, 1, 64

# Let reservoir run
op, res_states, Win, W, Wout = realesn(inSize, outSize, resSize)

# we'll disregard the first few states and solve for Wout
transient = 20000
extended_states = np.concatenate((test_ip, res_states),axis=0)
Wout = np.dot(test_op[transient:], 
               np.linalg.pinv(extended_states[:, transient:]))


# Compute the output after training
resop = np.dot(Wout, extended_states)
if resop.ndim == 1:
    resop.shape = (1,len(resop))
for index in range(outSize):
    result = Tseries(resop[index,transient:],tseries.time[transient:])
    result.plot(label = "ANN Output"+str(index))
plt.plot(tseries.time, test_op, label = 'Expected output')

# ---------------------------------
# Sigma Delta ESN transient spiking simulation - numpy
# ---------------------------------
def sdesn(Win, Wout, W):
    """
    Runs transient spiking simulation of an Echo-state 
    network given a set of Win, Wout, and W
    Win = Input connectivity matrix
    Wout = Readout connectivity matrix
    W = Recurrent connectivity matrix
    
    # Variables used in this function:
    # X = input: dim = (nip + 1) x 1
    # N = input neurons to convert X to spikes: dim = nip x 1
    # R = reservoir neurons: dim = nres x nres
    # Y = output neurons: dim = nop x 1
    # Wab = weight matrix such that B = W.A
    # Xip = input to the layer X
    # Xop = output from layer X
    """
    nItau    = 50e-12
    nIth     = 50e-12
    nfbItau  = 50e-12
    nfbIth   = 500e-12
    nfbIin   = 40e-9
    neItau   = 10e-12
    neIth    = 50e-9
    nnodes,nip,nnip,nnres,nop = compute_nodes(Win,Wout,W)
    X = test_ip
    N = Neuron(nnip,
             ipItau  = nItau,
             ipIth   = nIth,
             fbItau  = nfbItau,
             fbIth   = nfbIth,
             fbIin   = nfbIin,
             eItau   = neItau,
             eIth    = neIth,
             Ts=Ts)
    R = Neuron(nnres,
             ipItau  = nItau,
             ipIth   = nIth,
             fbItau  = nfbItau,
             fbIth   = nfbIth,
             fbIin   = nfbIin,
             eItau   = neItau,
             eIth    = neIth,
             Ts=Ts)
    Y = Neuron(nop,
             ipItau  = nItau,
             ipIth   = nIth,
             fbItau  = nfbItau,
             fbIth   = nfbIth,
             fbIin   = nfbIin,
             eItau   = neItau,
             eIth    = neIth,
             Ts=Ts)
    O = Vsyn(nop,
             Itau  = nItau,
             Ith   = nIth,
             Ts=Ts)    # Readout to LPF neuron output

    Wxn = np.eye(nip)
    Wnr = Win
    Wrr = W
    Wry = Wout
    Wyo = np.eye(nop)
    
    Rop = np.zeros(nnres)
    
    count = nop
    esn = np.zeros((count,len(tseries.time)))
    for index in range(len(tseries.time)):
        ip = X[:,index]
    
        Nip = np.dot(Wxn,ip)
        Nop = N.compute(Nip)
        
        Rip = np.dot(Wnr,Nop) + np.dot(Wrr,Rop)
        Rop = R.compute(Rip)
        
        Yip = np.dot(Wry,np.hstack((ip,Rop)))
        Yop = Y.compute(Yip)
        
        O.compute(Yop)
        esn[:,index] = O.states
    return esn,count

# Run spiking simulation
esn,count = sdesn(Win, Wout, W)

transient = 0
for index in range(count):
    esnres = Tseries(esn[index,transient:],tseries.time[transient:])
    esnres.plot(0, label = "SNN Output")
    esnres = Tseries(resop[index,transient:],tseries.time[transient:])
    esnres.plot(0, label = "Floating point ESN Output")
# plt.savefig("ANN-SDNN.pdf", dpi=300)
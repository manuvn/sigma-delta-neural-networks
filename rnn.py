import numpy as np
from spiker import Neuron

class rnn(object):
    def __init__(self, nrlayers, iunits, hunits, ounits, weights=[], nparams = {}):
        self.nrlayers = nrlayers
        self.iunits  = iunits
        self.hunits  = hunits
        self.ounits  = ounits
        if weights == []:
            self.weights = self.create_weights()
        else:
            self.weights = weights # add checker later

        self.nparams = {}
        # defaults
        self.nparams['ipItau']   = 50e-12
        self.nparams['ipIth']    = 50e-12
        self.nparams['fbItau']   = 50e-12
        self.nparams['fbIth']    = 50e-12
        self.nparams['fbIin']    = 4e-8
        self.nparams['eItau']    = 10e-12
        self.nparams['eIth']     = 50e-9
        self.nparams['thresh']   = 0.1e-9  
        self.nparams['Ts']       = 1e-6
        self.nparams['Kappa']    = 0.7 
        self.nparams['Temp']     = 300
        self.nparams['ipCmem']   = 2e-12
        self.nparams['fbCmem']   = 2e-12
        self.nparams['eCmem']    = 2e-12
        self.nparams['Tp']       = 1e-6
        self.nparams['mode']     = 'sd'
        self.nparams['refr']     = 1e-3
        self.nparams['mismatch'] = 0

        # if set by programmer
        for p in nparams.keys():
            if p in self.nparams.keys():
                self.nparams[p] = nparams[p]
            else:
                print('Incorrect key {}'.format(p))
        self.neurons = self.create_neurons()
        self.filter_factor = self.compute_ann_retention_ratio()

    def train_last_layer(self, tgt, ro_state):
        # we'll disregard the first few states and solve for Wout
        transient = 0
        Wout = np.dot(tgt[transient:], 
                    np.linalg.pinv(ro_state[:, transient:]))
        self.weights[-1] = Wout
        return

    def compute_ann_retention_ratio(self):
        Cmem  = self.nparams['fbCmem']
        Kappa = self.nparams['Kappa']
        Temp  = self.nparams['Temp']
        Itau  = self.nparams['fbItau']
        Ut    = 1.38064852e-23 * Temp/1.60217662e-19 # kt/q
        tau   = Cmem*Ut/(Kappa * Itau)
        a =  1 - np.exp(-self.nparams['Ts']/tau)
        self.tau = tau
        return a

    def create_neurons(self):
        neurons = []
        for _ in range(self.nrlayers):
            neurons +=  [Neuron(self.hunits, **self.nparams)]
            
        return neurons

    # create a network with n layers, input and output shape
    def create_weights(self):
        """
        Assumption is that all dot products will be done as np.dot(W, ip)
        nrlayers = number of recurrent layers
        nhid = number of neurons in recurrent layer
        nip = input dimension
        nop = output dimension
        """
        weights = []

        # input weight
        weights += [np.random.rand(self.hunits,self.iunits)-0.5]

        for _ in range(self.nrlayers):
            W = np.random.rand(self.hunits,self.hunits)-0.5
            # compute the spectral radius
            radius = np.max(np.abs(np.linalg.eigvals(W)))
            # rescale to desired spectral radius
            W = W * 1.4 / radius
            weights += [W]

            # feedforward weight
            W = np.random.rand(self.hunits,self.hunits)-0.5
            weights += [W]

        # output weight - overwrite the last weight to have op dimension
        weights[-1] = np.random.rand(self.ounits,self.hunits)-0.5
        return weights

    def ann_forward(self, ip, mode='ann'):
        """
        input shape = dim, length
        weights = weights of the layers of rnn
        a = time constant of the neurons in the network
        mode = spiking or non-spiking mode of operation
        """
        nlayers = len(self.weights)
        ip_len  = ip.shape[1]

        prev_state = None
        # create state variables to remember the intermediate states
        states = []
        for idx in range(nlayers):
            W = self.weights[idx]
            state_dim = W.shape[0]
            state = np.zeros((state_dim, ip_len))
            states += [state]

        prev_states = []
        for idx in range(self.nrlayers):
            prev_states += [None]

        # rnn forward pass
        for t_idx in range(ip_len):
            layer_ip = ip[:, t_idx]
            layer_op = np.dot(self.weights[0], layer_ip)
            states[0][:, t_idx] = layer_op

            for idx in range(self.nrlayers):
                fwd_idx = 2*idx+2
                rec_idx = 2*idx+1
                # First the recurrent calculation
                layer_op   = self.rec_layer(rec_idx, t_idx, layer_op, prev_states[idx], mode)
                prev_states[idx] = layer_op
                states[rec_idx][:, t_idx] = layer_op

                # Then, the forward pass
                layer_op     = np.dot(self.weights[fwd_idx], layer_op)
                states[fwd_idx][:, t_idx] = layer_op
        return states

    def rec_layer(self, rec_idx, t_idx, layer_ip, prev_state, mode):
        if t_idx > 0:
            layer_state = prev_state
        else:
            layer_op_dim = self.weights[rec_idx].shape[0]
            layer_state = np.zeros(layer_op_dim)

        layer_rec_ip = np.dot(self.weights[rec_idx], layer_state)
        rec_layer_ip = layer_rec_ip + layer_ip

        if mode == 'ann':
            rec_layer_ip = np.clip(rec_layer_ip, 0, self.nparams['fbIin'])# clamp + relu
            layer_op = (1-self.filter_factor)*layer_state + self.filter_factor*rec_layer_ip # filtering
        else:
            layer_idx = int(0.5*(rec_idx - 1))
            layer_op = self.neurons[layer_idx].compute(rec_layer_ip)
        return layer_op

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ip_dim = 1
    op_dim = 1
    res_dim = 1
    nrlayers = 1
    seq_len = 10

    net = rnn(nrlayers, ip_dim, res_dim, op_dim)

    # some random input
    ip = np.random.rand(ip_dim,seq_len)
    print('input', ip)
    print('='*89)

    print('ANN')
    print('='*89)
    ann_states = net.ann_forward(ip, mode='ann')
    for s in ann_states:
        print(s, '\n')

    print('SNN')
    print('='*89)
    snn_states = net.ann_forward(ip, mode='snn')
    for s in snn_states:
        print(s, '\n')

    for idx in range(len(ann_states)):
        plt.figure()
        plt.plot(ann_states[idx][0,:],'g')
        plt.plot(snn_states[idx][0,:],'k')

        plt.show()

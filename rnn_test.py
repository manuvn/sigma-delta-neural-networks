import numpy as np

# test module
from rnn import rnn

def test_1D_nofilt():
    ip_dim = 1
    op_dim = 1
    res_dim = 1
    nrlayers = 1
    seq_len = 10
    net = rnn(nrlayers, ip_dim, res_dim, op_dim)
    for idx in range(len(net.weights)):
        net.weights[idx][net.weights[idx] > 0] = 1
        net.weights[idx][net.weights[idx] < 0] = 1
        net.weights[idx][net.weights[idx] == 0] = 1
    net.weights[1] = np.array([[0]])

    # some random input
    ip = np.random.rand(ip_dim,seq_len)
    ip += 1
    states = net.ann_forward(ip)
    for s in states:
        if np.sum((s != ip) > 0):
            print('Error')

if __name__ == '__main__':
    test_1D_nofilt()
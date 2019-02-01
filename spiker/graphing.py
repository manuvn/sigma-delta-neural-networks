import re
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def compute_nodes(Win,Wout,W):
    #Structure of the network. Ip -> Vsyn -> Neuron. Neuron -> Syn -> Neuron, Neuron -> Syn -> Neuron
    # Win = nip x nneurons
    # W = nneurons x nneurons
    # Wout = nout x nneurons
    if np.ndim(Wout) == 1:
      Wout.shape = (1,len(Wout))
    nip = len(Win.T) # number of inputs
    nop = len(Wout) # number of outputs
    nnres = len(W) # number of neurons
    nnip = nip # number of virtual synapses
    nnodes = nip + nnip + nnres + nop
    return nnodes,nip,nnip,nnres,nop

def add_edges(G,ip,op,W):
    # op = W.in
    edgelist = []
    nrows,ncols = np.shape(W)
    for r in range(nrows):
        for c in range(ncols):
            if W[r,c]:
                # Only create an edge if weight is not zero
                edge = [(ip[r],op[c],W[r,c])]
                edgelist = edgelist + edge
    G.add_weighted_edges_from(edgelist)
    return G

def create_conn_matrix(W,Win,Wout):
    # refer to diagram for illustration or image titled matrix.jpg
    nnodes,nip,nnip,nnres,nop = compute_nodes(Win,Wout,W)
    full_W = np.zeros((nnodes,nnodes))
    full_W[nip:nip+nnip,0:nip] = np.eye(nip)
    full_W[nip+nnip:nip+nnip+nnres,nip:nip+nnip] = Win
    full_W[nip+nnip:nip+nnip+nnres,nip+nnip:nip+nnip+nnres] = W
    full_W[nip+nnip+nnres:,nip:nip+nnip] = Wout[:,:nip] # Input is connected to output
    full_W[nip+nnip+nnres:,nip+nnip:nip+nnip+nnres] = Wout[:,nip:]

    return full_W

def draw_graph(G,Win,Wout,W,names,icolor='r',vcolor='g',rcolor='y',ocolor='m'):

    # fix input and output node positions
    nnodes,nip,nnip,nnres,nop = compute_nodes(Win,Wout,W)
    print("Total nodes # = ",nnodes)
    print("Input nodes # = ",nip)
    print("Input neuron # = ", nnip)
    print("Reservoir neuron # = ",nnres)
    print("Output neuron # = ", nop )
    f = 1.0
    op_xindex, ip_xindex, nip_xindex = f*0.99,f*0.01,f*0.2
    res_radius, res_xcenter, res_ycenter  = f*0.2, f*0.6, f*0.5
    op_pos,ip_pos,nin_pos = f/(nop+1),f/(nip+1),f/(nnip+1)
    op_step, ip_step, nin_step = f/(nop+1),f/(nip+1),f/(nnip+1)
    fixed_positions = {}
    labels={}
    nodecolor = []
    for name in G.nodes():
        labels[name]=name
        if re.match(r'x', name):
            fixed_positions[name] = (ip_xindex,ip_pos)#dict with two of the positions set
            ip_pos = ip_pos + ip_step
            color = icolor
        elif re.match(r'y', name):
            fixed_positions[name] = (op_xindex,op_pos)#dict with two of the positions set
            op_pos = op_pos + op_step
            color = ocolor
        elif re.match(r'n', name):
            fixed_positions[name] = (nip_xindex,nin_pos)#dict with two of the positions set
            nin_pos = nin_pos + nin_step
            color = vcolor
        else:
            r, theta = [np.random.uniform(0.0,res_radius), 2*np.pi*np.random.random()]
            x = res_xcenter + r * np.cos(theta) 
            y = res_ycenter + r * np.sin(theta)
            fixed_positions[name] = (x,y)#dict with two of the positions set        
            color = rcolor
            labels[name]='.'
        nodecolor = nodecolor+[color]
    fixed_nodes = fixed_positions.keys()
    npos = nx.spring_layout(G,pos=fixed_positions, fixed = fixed_nodes)
    nx.draw_networkx_nodes(G,npos,node_color=nodecolor, node_size=1000,alpha=0.6)
    nx.draw_networkx_edges(G,npos,width=1.0,alpha=0.6)
    nx.draw_networkx_labels(G,npos,labels,font_size=12)
    plt.axis('off')

def create_graph_nodes(W,Win,Wout):
    nnodes,nip,nnip,nnres,nop = compute_nodes(Win,Wout,W)
    nodenames = []
    for index in range(nnodes):
        if index <  nip:
            indexstr = str(index)
            name = 'x'+ indexstr
        elif index < nip + nnip:
            indexstr = str(index - nip)
            name = 'n'+ indexstr
        elif index < nip + nnip + nnres:
            indexstr = str(index - nip - nnip)
            name = 'r'+ indexstr
        else:
            indexstr = str(index - nip - nnip - nnres)
            name = 'y'+ indexstr
        nodenames = nodenames+[name]
        
    # Create graph object
    Gn=nx.Graph(name="Connectivity")
    Gn.add_nodes_from(nodenames)
    return Gn,nodenames

def create_graph(W,Win,Wout,icolor='r',vcolor='g',rcolor='y',ocolor='m'):
    G,names = create_graph_nodes(W,Win,Wout)
    full_W = create_conn_matrix(W,Win,Wout)
    G = add_edges(G,names,names,full_W)
    draw_graph(G,Win,Wout,W,names,icolor,vcolor,rcolor,ocolor)
    return G,full_W

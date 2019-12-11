from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from sklearn.linear_model import SGDRegressor
from progressbar import progressbar


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
Sets up the environment

Variations from original work:

Attacker chooses attack intensity: 1,2.  Probability of detection scales with attack intensity: intensity*base_probability
Attacker moves from node to node and can only act on nodes adjacent to he one it is in. 
'''

def onehot(c, n):
    a = np.zeros(n)
    a[c] = 1
    return a


class Network:
    def __init__(self, n_nodes, v_per_node, A = None, data_node = 3, start_node = 0, base_detect_prob = 0.1, initial_security = 2):
        
        self.A = A

        if self.A is None:
            self.A = np.array([0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]).reshape(4,4) #adjacency matrix for graph
        
        self.initial_security = initial_security
        self.V = np.zeros((n_nodes,v_per_node)) +initial_security #record security level per vulnerability for nodes. Initilize with max security on all but 1. subject to change
        self.V[start_node]-=initial_security
        
        
        self.vulnerabilities = np.array(np.zeros(shape=(n_nodes,), dtype = 'int32'))+0#np.random.randint(v_per_node, size = n_nodes) #np.random.randint(v_per_node, size = n_nodes)

        #print(self.vulnerabilities)

        for i in range(1,n_nodes):
            v = self.vulnerabilities[i]
            self.V[i][v]=0
        
        self.detect = np.zeros(n_nodes) +base_detect_prob # probability of attacker detection per node
        self.detect[start_node] = 0
        self.cracked = np.zeros(n_nodes) # records whether a node has been cracked 
        self.cracked[start_node] = 1 
        self.start_node = start_node
        self.data_node = data_node
        self.n_nodes = n_nodes
        self.v_per_node = v_per_node
        self.detected_attacker = False
       
       
    def reset(self):
        self.cracked = np.zeros(self.n_nodes) # records whether a node has been cracked 
        self.cracked[self.start_node] = 1 
        self.detected_attacker = False
        self.V = np.zeros((self.n_nodes,self.v_per_node)) + self.initial_security#record security level per vulnerability for nodes. Initilize with max security on all but 1. subject to change
        
        self.V[self.start_node] = 0
        for i in range(1,self.n_nodes):
            v = self.vulnerabilities[i]
            self.V[i][v]=0

    def change_v(self,node,v,c):
        self.V[node][v] +=c

    

    

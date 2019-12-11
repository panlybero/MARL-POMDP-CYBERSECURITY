import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from sklearn.linear_model import SGDRegressor
from network import *

class Defender:
    '''

    Makes Q estimates based on entire state of the network. 

    '''
    def __init__(self, network, linear = False, learning = "sarsa", feature_set = "full", defend = True):

        self.attacker_node = network.start_node
        self.linear = linear
        self.Qnet = None
        self.feature_set = feature_set
        self.reward = 0
        self.network = network
        self.f = 0
        self.defense_val = 1
        if not defend:
            self.defense_val = 0

        if self.feature_set == 'full':
            self.perc_act_size = network.n_nodes * network.v_per_node  #full security of network
        if self.feature_set == 'slim':
            self.perc_act_size = network.n_nodes # security level of vulnerability per node

        self.learning = learning

        if not linear:
            inputs = Input(shape= (self.perc_act_size,))

            h1 = Dense(units = 10, activation = 'relu')(inputs)

            y = Dense(units= 1, name = 'output', activation= 'linear')(h1)

            self.Qnet = tf.keras.Model(inputs = inputs, outputs = y)
            self.Qnet.compile(loss='mse',
                    optimizer='adam',
                    metrics=['mse'])
            
            init_x = np.random.random(size = self.perc_act_size).reshape(1,-1)
            init_y = np.random.random(1).reshape(-1,1)
            self.Qnet.fit(init_x,init_y, epochs = 1, verbose = 0)
        else:
            self.Qnet =SGDRegressor(loss='squared_loss', max_iter = 5, learning_rate='constant',eta0=0.1)
            
            self.Qnet.partial_fit(np.random.random(size = self.perc_act_size).reshape(1,-1)*10,np.random.random(size=1).reshape(-1)*10)
            self.Qnet.coef_ = np.zeros(self.perc_act_size)
            self.Qnet.intercept_ = np.zeros(1)
            
        return

    def reset(self):
        self.attacker_node = self.network.start_node
    
    def observe(self, network, node = None):
        if self.feature_set == "full":
        
            w = network.V.reshape(-1).copy()

            w[np.where(w == 2)]  = 2
            
            return w

        elif self.feature_set == "slim":
            o = []
            for i in range(network.n_nodes):
                o.append(min(network.V[i]))
            return np.array(o)


        
    def get_weights(self):
        return self.Qnet.coef_

    def make_obs_action_pairs(self,actlist, network): # what the network security will look like
        if self.feature_set == 'full':
            o = self.observe(network)
            observations = []
            for i in actlist:
                tmp = o.copy()
                if tmp[i] == 1:
                    tmp[i] = 2
                if tmp[i] == 0:
                    tmp[i] = 1
                
                observations.append(tmp)
            
            return np.array(observations)
        
        elif self.feature_set == 'slim':
            o = self.observe(network)
            observations = []
            for i in actlist:
                tmp = o.copy()
                if tmp[i] == 1:
                    tmp[i] = 2
                if tmp[i] == 0:
                    tmp[i] = 1
                
                observations.append(tmp)
            
            return np.array(observations)


    def select_action(self, network, e):
        
        actlist = None
        if self.feature_set == 'full':
            actlist = list(range(network.v_per_node*network.n_nodes))
        elif self.feature_set == 'slim':
            actlist = list(range(network.n_nodes))
        pairs = self.make_obs_action_pairs(actlist,network)
        
        x = np.random.random(1)
        if x<=e:
            
            action = np.random.randint(self.perc_act_size)
            return action
        else:
            vals = self.Qnet.predict(pairs)
            action = np.argmax(vals)
            
            
            
            return action
        
    def set_reward(self,r):
        self.reward = r

    def do_action(self, network, action): #return reward

        if self.feature_set == 'full':
            node = int(np.floor(action/network.v_per_node))
            #print(node,action)
            v = action % network.v_per_node
           
            network.V[node][v] = min(2,network.V[node][v]+self.defense_val)
            self.set_reward(0)
            if np.sum(network.V[1:]) == (network.n_nodes-1)*network.v_per_node*2:
                #print("all")
                self.set_reward(100)
            return 
        
        elif self.feature_set == 'slim':
            v = np.argmin(network.V[action])
            network.V[action][v] = min(2,network.V[action][v]+self.defense_val)

            self.set_reward(0)
            if np.sum(network.V[1:]) == (network.n_nodes-1)*network.v_per_node*2:
                #print("all")
                self.set_reward(100)
            return 


    def QsarsaUpdate(self,sarsa, gamma):
        perc_act,reward,next_perc_act = sarsa
        a = 0.1
        #print(perc_act)
        curr = self.Qnet.predict(perc_act.reshape(1,-1))
        next_ = self.Qnet.predict(next_perc_act.reshape(1,-1))

        ret = np.array([curr + a*(reward + gamma*next_ -curr)]).reshape(-1)

        if self.linear:
            self.Qnet.partial_fit(perc_act.reshape(1,-1), ret)
        else:
            self.Qnet.fit(perc_act.reshape(1,-1), ret, verbose = 0)
    

    def _calculate_returns(self,sarsas,gamma, alpha= 1):

        all_rets = []
        s = 0
        _sarsas = sarsas.copy()
        _sarsas.reverse()
        x = []
        for sarsa in _sarsas:
            perc_act,reward,next_perc_act = sarsa
            s= 0
            if len(all_rets)>0:
                s = alpha*(gamma*all_rets[-1] + reward)
            else:
                s = reward
            all_rets.append(s)
            x.append(perc_act)

        all_rets.reverse()
        x.reverse()
        return np.array(all_rets), np.array(x)

    def QMonteCarlo(self,sarsas, gamma):

        returns,x = self._calculate_returns(sarsas,gamma)
        
        
        if self.linear:
            self.Qnet.partial_fit(x, returns)
        else:
            self.Qnet.fit(x, returns, verbose = 0)
        
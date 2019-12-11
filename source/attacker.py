import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from sklearn.linear_model import SGDRegressor
import os
from network import *
from defender import *

class Attacker:
    '''

    Makes Q estimates based on index of node (starts at 0) and one-hot*intensity encoding of actions.
    Low intensity is 1, High intensity is 2 or -1. Note on implementation. if action is attack (node,3) with high intensity,
    the game will use [0 0 0 2] to decide if cracked, but the agent will see [0 0 0 intensity] where intensity is 2 or -1 (depending on symmetric or incremental setting).
    Thsi fascilitates learning, especially with linear models, as the model does not get confused by the incremental space.

    '''
    def __init__(self, network, linear = False, intensity_mode = 'incremental', learning = "sarsa", defender = None, feature_set = 'full', static= False):

        self.defender = defender
        self.current = network.start_node
        self.linear = linear
        self.Qnet = None
        self.featureset = feature_set
        self.static = static
        if feature_set == 'full':
            self.perc_act_size = network.n_nodes+network.v_per_node #1+network.v_per_node  #network.n_nodes+network.v_per_node
        elif feature_set == 'slim':
             self.perc_act_size = network.n_nodes+1 #onehot node plus defense value of each node
        self.intensity_mode = intensity_mode
        self.network = network
        self.learning = learning

        if not linear:
            inputs = Input(shape= (self.perc_act_size,))

            h1 = Dense(units = 10, activation = 'relu')(inputs)

            y = Dense(units= 1, name = 'output', activation= 'linear')(h1)

            self.Qnet = tf.keras.Model(inputs = inputs, outputs = y)
            self.Qnet.compile(loss='mse',
                    optimizer='sgd',
                    metrics=['mse'])
            
            init_x = np.random.random(size = self.perc_act_size).reshape(1,-1)
            init_y = np.random.random(1).reshape(-1,1)
            self.Qnet.fit(init_x,init_y, epochs = 1, verbose = 0)
        else:
            self.Qnet =SGDRegressor(loss="squared_loss", max_iter = 5, learning_rate='constant', eta0=0.1)
            self.Qnet.partial_fit(np.random.random(size = self.perc_act_size).reshape(1,-1)*10,np.random.random(size=1).reshape(-1)*10)
            self.Qnet.coef_ = np.zeros(self.perc_act_size)
            self.Qnet.intercept_ = np.zeros(1)
        return

    def reset(self):
        self.current = self.network.start_node
    
    def observe(self, network, node = None):
        
        if node==None:
            node = self.current

        node_onehot = np.array(onehot(node,self.network.n_nodes)) #network.A[node] #observes accesible nodes from node
        return node_onehot
        
    def get_weights(self):
        return self.Qnet.coef_

    def make_obs_action_pairs(self,neighbors,actlist, network):
        observations = [] 
        for i in neighbors:
            o = self.observe(network, node = i)
            if self.featureset == 'full':
                observations.append(o)

            elif self.featureset == 'slim':
                for j in range(network.v_per_node*2):
                    observations.append(o)

        actions = []
        if self.featureset == 'full':
            for i in actlist:
                oh = onehot(i,len(actlist))
                actions.append(oh)
                if self.intensity_mode == 'incremental': #high intensity is 2 if incremental or -1 if symmetric. 
                    actions.append(oh*2)
                elif self.intensity_mode == 'symmetric':
                    actions.append(oh*-1)
        elif self.featureset == 'slim':
            for n in neighbors:    
                for v in range(network.v_per_node):
                    for inten in range(2):
                        c = inten%2 +1
                        a = c-network.V[n][v]
                        actions.append([a])
        
        pairs = []

        if self.featureset == 'full':
            for obs in observations:
                for acts in actions:
                    pairs.append(np.concatenate((obs,acts)))
            pairs = np.array(pairs)
        elif self.featureset == 'slim':
            for i in range(len(observations)):
                pair = np.concatenate((observations[i], actions[i]))
                pairs.append(pair)
            pairs = np.array(pairs)

        
        return pairs

    def select_intensity(self, action = None):
        if action is None:
            if self.intensity_mode == 'incremental':
                return np.random.randint(1,2)
            elif self.intensity_mode == 'symmetric':
                return np.random.choice([-1,1])
        else:
            tmp = {1:1,-1:2, 2:-1}
            if self.intensity_mode == 'incremental':
                return sum(action)
            elif self.intensity_mode == 'symmetric':
                return tmp[sum(action)]



    def select_action(self, network, e):
        adj = network.A[self.current]#self.observe(network) #gets neighbors of current node
        neighbors = np.where(adj==1)[0]
        #print(neighbors)
        actlist = list(range(network.v_per_node)) 
        pairs = self.make_obs_action_pairs(neighbors,actlist,network)
        
        x = np.random.random(1)
        if x<=e:
            node = np.random.randint(network.n_nodes)#np.random.choice(neighbors)
            
            action = np.random.choice(actlist)
            
            intensity = self.select_intensity()
            if self.featureset == 'full':
                action = onehot(int((action-1)/2), network.v_per_node)*intensity
               
            elif self.featureset == 'slim':
                action =  np.random.randint(network.v_per_node)
                inten = np.random.randint(2)
                action = (action,inten)
            
            return (node,action)

            
        else:
            if self.featureset == 'full':
                vals = self.Qnet.predict(pairs)
                vals = vals.reshape(-1,network.v_per_node*2)
                a = np.unravel_index(np.argmax(vals, axis=None), vals.shape)
                
                if(self.static):
                    if np.random.random(1)>0.5:
                        a = (1,onehot(2, network.v_per_node))
                    else:
                        a =(3,onehot(2, network.v_per_node))
                    return a
                


                node =neighbors[a[0]]
                
                action =None
            
                if a[1] % 2 == 0:
                    action = onehot(int(a[1]/2), network.v_per_node)
                else:
                    
                    action = onehot(int((a[1]-1)/2), network.v_per_node)*2
                
                
                return (node,action)

            if self.featureset == 'slim':
                
                vals = self.Qnet.predict(pairs)
                
                a = np.unravel_index(np.argmax(vals, axis=None), vals.shape)
                k = len(vals)/len(neighbors)
                
                node = int(a[0]/k)
                v = a[0] % network.v_per_node
                inten = a[0] % 2  +1
                #node =neighbors[a[0]]
                action = (v,inten)
                r = (node,action)
                
                
                return r


    
            
            return (node,action)
        

    def do_action(self, network, node, action): #return reward

        vuln = None
        a = None
        attack_val = None
        def_val = None

        if self.featureset == 'full':
            vuln = network.V[node]
            a = np.where(action!=0)[0]
            attack_val = action[a]
            def_val = vuln[a]
            intensity = self.select_intensity(action=action)
        
        if self.featureset == 'slim':
            vuln = network.V[node]
            attack_val = action[1]
            def_val = vuln[action[0]]
            intensity = action[1]


        #print(action,vuln,a)
        
        if attack_val > def_val: # Successful attack?
            
            network.cracked[node] = 1

            #intensity = self.select_intensity(action=action)
            #print(intensity)
            detect_prob = network.detect[node] * intensity
           
            self.current = node
            
            if node == network.data_node: #cracked target node, end episode
                network.cracked[network.data_node] = 1
                self.defender.set_reward(-100)
                return 100
            else:
                if np.random.random(1)<= detect_prob: #detected?
                    network.detected_attacker= True
                    self.defender.set_reward(100)
                    return -100 # end episode, punish attacker

                return -1 #not detected
        else:
            return -1 #unsuccessful attack

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
    

    def _calculate_returns(self,sarsas,gamma,alpha=1):

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
                s = alpha*reward
            all_rets.append(s)
            x.append(perc_act)

        all_rets.reverse()
        x.reverse()
        return np.array(all_rets), np.array(x)

    def QMonteCarlo(self,sarsas, gamma):
       
        returns,x = self._calculate_returns(sarsas,gamma)
        #returns = returns.reshape(-1,1)
        
        

        
        if self.linear:
            self.Qnet.partial_fit(x, returns)
        else:
            self.Qnet.fit(x, returns, verbose = 0)
        
    


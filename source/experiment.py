import numpy as np

from attacker import *
from defender import *
from network import *
from tqdm import tqdm
import os



def check_episode_end(network):
    if network.cracked[network.data_node] == 1 or network.detected_attacker:
        return True

    else:
        if np.sum(network.V[1:]) == (network.n_nodes-1)*network.v_per_node*2:
            
            return True
        else:
            return False
        return False

    


def episode(network,attacker,defender = None, length = -1, e = 0.1, train = True):


    rewards = []

    network.reset() # Need to write reset functions
    attacker.reset()
    defender.reset()


    a = attacker.select_action(network,e)
    

    o = np.array([a[0]])
    #network.A[a[0]]
   
    perc_act = np.concatenate((o,a[1]))
    sarsas = []
    
    i = 0
    while True:
        
        i+=1
        r = attacker.do_action(network,a[0],a[1])
        rewards.append(r)

        a = attacker.select_action(network,e)
        o = np.array([a[0]])
        #network.A[a[0]]
        #print(perc_act)
        next_perc_act = np.concatenate((o,a[1]))
        

        sarsa = (perc_act,r,next_perc_act)
        sarsas.append(sarsa)

        if train and attacker.learning == 'sarsa':
            attacker.QsarsaUpdate(sarsa,0.95)
        
        perc_act = next_perc_act

        if check_episode_end(network) or i==length:
            break

    if train and attacker.learning == 'montecarlo':

        attacker.QMonteCarlo(sarsas,0.95)

    if not train:
        return rewards
    return sum(rewards)


defender_acts = []
defender_nodes = []
attacker_acts = []
attacker_nodes =[]
def episode_both(network,attacker,defender, length = -1, e = 0.1, train = True):


    rewards = []
    rewards_d = []

    network.reset() # Need to write reset functions
    attacker.reset()
    defender.reset()
    egr = e
    if train is False:
        egr = 0

    a = attacker.select_action(network,egr)
    
    act = None
    if attacker.featureset == 'full':
        act = a[1]
    elif attacker.featureset == 'slim':
        act = np.array([a[1][1]-network.V[a[0]][a[1][0]]])


    a_d = defender.select_action(network,egr)
    
    o = np.array(onehot(a[0],network.n_nodes))
    o_d = defender.observe(network)
    
    o_d[a_d] = min(2, o_d[a_d]+1)
    defender_nodes.append(int(a_d))
    perc_act = np.concatenate((o,act))
    perc_act_d = o_d
    sarsas = []
    sarsas_d = []
    
    i = 0
    while True:
        
        i+=1
        defender.do_action(network,a_d)
        #print(a)
        #os.system("pause")
        r = attacker.do_action(network,a[0],a[1])
        rewards.append(r)
        rewards_d.append(defender.reward)
        
        a_d = defender.select_action(network,egr)
        o_d = defender.observe(network)
        o_d[a_d] = min(2, o_d[a_d]+1)
        next_perc_act_d = o_d
        defender_acts.append(a_d)


        a = attacker.select_action(network,egr)
        attacker_acts.append(a[0]*4+np.where(a[1]!=0)[0][0])
        attacker_nodes.append(attacker.current)
        
        o = onehot(a[0],network.n_nodes)

        act = None
        if attacker.featureset == 'full':
            act = a[1]
        elif attacker.featureset == 'slim':
            act = [a[1][1]-network.V[a[0]][a[1][0]]]
        
        
        #network.A[a[0]]
        #print(perc_act)
        
        next_perc_act = np.concatenate((o,act))
        

        sarsa = (perc_act,r,next_perc_act)
        
        sarsa_d = (perc_act_d,defender.reward,next_perc_act_d)
       
       
        #print(sarsa_d)
        sarsas.append(sarsa)
        sarsas_d.append(sarsa_d)

        if train and attacker.learning == 'sarsa':
            attacker.QsarsaUpdate(sarsa,0.95)
        
        perc_act = next_perc_act
        perc_act_d = next_perc_act_d

        if check_episode_end(network) or i==length:
            break
    
    #print('ep end')
    if train and attacker.learning == 'montecarlo':

        attacker.QMonteCarlo(sarsas,0.5)

    if train and defender.learning == 'montecarlo':
        defender.f+=1
        defender.QMonteCarlo(sarsas_d,0.5)
    
   
    
    
    if not train:
        return rewards,rewards_d
    return sum(rewards),sum(rewards_d)


import matplotlib.pyplot as plt

            
if __name__ == '__main__':

    np.random.seed(seed = 222)

    
    
    
    graph = nx.chvatal_graph()
    A = nx.to_numpy_array(graph)
    A = np.array([0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]).reshape(4,4)
    '''
    A = np.array([0,1,0,0,
                  1,0,1,0,
                  0,1,0,1,
                  0,0,1,0]).reshape(4,4)
    
    A = np.array([0,1,1,0,0,
                  1,0,1,0,0,
                  1,1,0,1,0,
                  0,0,1,0,1,
                  0,0,0,1,0]).reshape(5,5)
    '''
    n_agents = 1
    
    print(A)

    episodes = 500

    avg = np.zeros(episodes)
    avg_d = np.zeros(episodes)
    attacker = None
    
    for agents in tqdm(range(n_agents)):
        network = Network(len(A),4, A = A, base_detect_prob= 0.1, data_node=3, initial_security=1)
        
        defender = Defender(network,linear = False, learning="montecarlo", feature_set="full", defend= True)
        attacker = Attacker(network, defender= defender, linear = False, intensity_mode='incremental', learning='montecarlo', feature_set='full',static = False)

        total_rewards = []
        total_rewards_d = []
        for i in tqdm(range(episodes)):
            r,r_d = episode_both(network,attacker, defender, length=100)
            total_rewards.append(r)
            total_rewards_d.append(r_d)
        total_rewards = np.array(total_rewards)
        total_rewards_d = np.array(total_rewards_d)
        avg+=total_rewards
        avg_d+=total_rewards_d

    avg/=n_agents
    avg_d/=n_agents
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    
    ax.scatter(range(len(avg)),avg, s = 2 )
    ax.scatter(range(len(avg)),avg_d, s = 2)


    wins = np.array(avg>avg_d, dtype='int32')
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.hist(wins,weights=np.ones(len(wins)) / len(wins))

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.hist(defender_acts,range=(0,17),bins=list(range(17)), color='red')
    
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    ax.hist(attacker_acts)

    fig = plt.figure(5)
    ax = fig.add_subplot(111)
    ax.scatter(range(len(attacker_acts)),attacker_acts, s = 2 )

    fig = plt.figure(6)
    ax = fig.add_subplot(111)
    colors = ["red","blue","green","black","yellow"]

    defender_nodes = np.array(defender_nodes)
    '''
    defender_acts = np.array(defender_acts)
    defender_acts = np.mod(np.array(defender_acts), len(A))
    cols = [colors[i] for i in defender_acts]
    for i in range(len(defender_nodes)):
        ax.scatter(i,defender_nodes[i], s = 4, color=cols[i])
    '''
    fig = plt.figure(7)
    ax = fig.add_subplot(111)
    ax.scatter(range(len(attacker_nodes)),attacker_nodes, s = 2 )


    fig = plt.figure(8)
    graph = nx.from_numpy_array(A)
    nx.draw_networkx(graph)
    plt.show()
    #r,r_d = episode_both(network,attacker, defender, length=-1)
    #print(r,r_d)

    
    

    
    
    
    
    
    






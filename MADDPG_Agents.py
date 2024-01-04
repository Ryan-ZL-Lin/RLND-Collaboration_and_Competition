import numpy as np
import random
import copy
import torch
import Dict_Hyperparams as P

from DDPG_Agent import Agent, ReplayBuffer
from collections import namedtuple, deque

import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    """MADDPG Agent : Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize a MADDPG Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        
        super(MADDPG, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        
        # Instantiate Memory replay Buffer (shared between agents)
        self.memory = ReplayBuffer(action_size, P.BUFFER_SIZE, P.BATCH_SIZE, random_seed)
        
        
        # Instantiate Multiple  Agent
        self.agents = [ Agent(state_size,action_size, random_seed, self.memory) 
                       for i in range(num_agents) ]
                  
    def reset(self):
        """Reset all the agents"""
        for agent in self.agents:
            agent.reset()

    def act(self, states):
        """Return action to perform for each agents (per policy)"""        
        return [ agent.act(state) for agent, state in zip(self.agents, states) ]
                
    
    def step(self, states, actions, rewards, next_states, dones):
        """ # for each agent, save experience in the shared replay memory, and use random sample from buffer to learn"""
        [self.agents[posit].step(states[posit], actions[posit], rewards[posit], next_states[posit], dones[posit]) for posit in range(self.num_agents)]
                                    
    def save_checkpoints(self, name=""):
        """Save checkpoints for all Agents"""
        for idx, agent in enumerate(self.agents):
            actor_local_filename = 'checkpoint_actor_local_' +str(name)+"_"+ str(idx) + '.pth'
            critic_local_filename = 'checkpoint_critic_local_' +str(name)+"_"+ str(idx) + '.pth'           
            actor_target_filename = 'checkpoint_actor_target_' +str(name)+"_"+ str(idx) + '.pth'
            critic_target_filename = 'checkpoint_critic_target_' +str(name)+"_"+ str(idx) + '.pth'            
            torch.save(agent.actor_local.state_dict(), actor_local_filename) 
            torch.save(agent.critic_local.state_dict(), critic_local_filename)             
            torch.save(agent.actor_target.state_dict(), actor_target_filename) 
            torch.save(agent.critic_target.state_dict(), critic_target_filename)

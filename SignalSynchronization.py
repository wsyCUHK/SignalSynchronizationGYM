
import gym
from gym import spaces
import numpy as np



class Signal:

    def __init__(self, value, velocity):
        self.value = value
        self.velocity = velocity
        self.value_trace = np.array([value])
        self.childlist = []
        self.parent = None


    def move(self):
        self.value += self.velocity
        self.value_trace =  np.append(self.value_trace,self.value)
        
    
    def AttachChild(self,child):
        child.parent = self
        self.childlist.append(child)
    
    def AttachToParent(self,parent):
        parent.AttachChild(self)
    

class Noise:
    def __init__(self,dist_name,parameter):
        self.dist_name = dist_name
        self.parameter = parameter
    
    def make_noise(self):
        
        if self.dist_name == 'Normal':
            noise = np.random.normal(self.parameter[0], self.parameter[1], 1)
            return noise[0]
        
        if self.dist_name == 'Exponential':
            noise = np.random.exponential(self.parameter[0],1)
            return noise[0]
        
        if self.dist_name == 'Poisson':
            noise = np.random.poisson(self.parameter[0],1)
            return noise[0]
        
        if self.dist_name == 'Uniform':
            noise = np.random.uniform(self.parameter[0],self.parameter[1],1)
            return noise[0]    



class SignalSync_Env(gym.Env):
    """
    num - number of signals
    
    epsilon - threshold for decide wether 2 signal are syncrhonized
    
    Obervasion (num X 1)
    observasion[i]:
    0                                                                - reset & root node
    measured difference between signal i-1 and its parent signal     - otherwise
    
    action space (num x 2)
    action[i][0] - value adjust for signal i-1
    action[i][1] - velocity adjust for signal i-1
    
    rewards (num X 1)
    reward[i]:    
    1 - if signal i-1 is synchronized to its parent
    0 - otherwise
 
    """ 
    def __init__(self, signal_list, noise_type, noise_para, epsilon):
        self.signal_list = signal_list
        self.num = len(self.signal_list)
        self.observation = np.zeros(self.num)
        self.noise = Noise(noise_type,noise_para)
        self.epsilon = epsilon
        self.reset()


    def AddSignal(self,signal):
        self.signal_list.append(signal)
        self.num = len(self.signal_list)
        self.reset()
    
    
    
    def SetNoise(self,noise_type,noise_para):
        self.noise = Noise(noise_type,noise_para)
    
    def SetEpsilon(self,epsilon):
        self.epsilon = epsilon
    
    
    def reset(self):
        self.observation = np.zeros(self.num)
        return self.observation
    
    
    def step(self, action):
    
        for i in range(self.num):
            if self.signal_list[i].parent != None:
                self.signal_list[i].value += action[i][0] + (action[i][0]!=0)*self.noise.make_noise()
                self.signal_list[i].velocity += action[i][1] + (action[i][1]!=0)*self.noise.make_noise()
            self.signal_list[i].move()
        
        # observation
        for i in range(self.num):
            if self.signal_list[i].parent != None:
                self.observation[i] = self.signal_list[i].value - self.signal_list[i].parent.value + self.noise.make_noise()
            else:
                self.observation[i] = 0
                
        done = False
        reward = np.zeros(self.num)
        
        # rewards
        for i in range(self.num):
            if self.signal_list[i].parent != None:
                if -self.epsilon <= self.signal_list[i].value - self.signal_list[i].parent.value <= self.epsilon:
                    reward[i] = 1
                else:
                    reward[i] = 0
            else:
               reward[i] = 1
                
        if np.array_equal(reward,np.ones(self.num)):
            done = True
            
        return self.observation, reward, done


class SignalSync_process:
    
    def __init__(self,simpy_env, signal_sync_env,time_interval):
        self.signal_sync_env = signal_sync_env
        self.signal_list = signal_sync_env.signal_list
        self.num = len(self.signal_list)
        self.time_interval = time_interval
        self.env = simpy_env
        self.action = simpy_env.process(self.run())
        self.total_reward = np.zeros(self.num)
        self.policy = DefaultPolicy()


    def SetPolicy(self, Policy):
        self.policy = Policy
    
    def run(self):
        action  = np.zeros((self.num,2))
        count = 0
        while True:
            yield self.env.timeout(self.time_interval)
            count+=1
            observation, reward, done = self.signal_sync_env.step(action)
            if done == True:
                print('Syncrhonized at iteration %d' %count )
            self.total_reward += reward
            
            """
            run policy
            """
            self.policy.Run(observation,reward,action)
            
            
class DefaultPolicy(object):
    def Run(self,observation,reward,action):
        num = len(observation)
        for i in range(num):
            if reward[i] == 1:
                action[i][0] = 0
                action[i][1] = 0
            if reward[i] == 0:
                action[i][0] = -observation[i]
                action[i][1] = -observation[i]
        return action

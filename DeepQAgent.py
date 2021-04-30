# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:03:43 2021

@author: Runar
"""


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.optimizers as kOptimizers
import numpy as np
import tensorflow.keras.models as kModel
import time

#print(tf.__version__)
#input()
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


#tf.config.gpu_options.allow_growth = True

class DeepQNetworkConv(keras.Model):
    def __init__(self,nActions,inputShape):
        super(DeepQNetworkConv,self).__init__()
        self.nActions = nActions
        self.inputShape = inputShape
        self.convLayer1 = keras.layers.Conv1D(filters=10,kernel_size=5,strides=1,input_shape=inputShape,activation='relu',dtype=tf.float32)
        self.convLayer2 = keras.layers.Conv1D(filters=20,kernel_size=2,strides=1,activation='relu')
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256)
        self.actL1 = keras.layers.LeakyReLU(alpha=0.7)
        self.fc2 = keras.layers.Dense(128)
        self.actL2 = keras.layers.LeakyReLU(alpha=0.7)
        self.actionLayer = keras.layers.Dense(nActions,activation='linear')
        
    def call(self,state):
        x = self.convLayer1(state)
        x = self.convLayer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.actL1(x)
        x = self.fc2(x)
        x = self.actL2(x)
        action = self.actionLayer(x)
        return action
    
    def get_config(self):
        return {"nActions":self.nActions,"inputShape":self.inputShape}
        

class DeepQNetwork(keras.Model):
    def __init__(self,nActions,inputShape):
        super(DeepQNetwork,self).__init__()
        self.nActions = nActions
        self.inputShape = inputShape
        self.inputLayer = keras.layers.Flatten(input_shape=inputShape,dtype=np.int32)
        self.denseLayer1 = keras.layers.Dense(256,activation='LeakyReLU')
        self.denseLayer2 = keras.layers.Dense(256,activation='LeakyReLU')
        self.denseLayer3 = keras.layers.Dense(128,activation='LeakyReLU')
        #self.V = keras.layers.Dense(1,activation=None)
        self.actionLayer = keras.layers.Dense(nActions,activation='linear')
    
    def call(self,state):
        x = self.inputLayer(state)
        x = self.denseLayer1(x)
        x = self.denseLayer2(x)
        x = self.denseLayer3(x)
        #V = self.V(x)
        action = self.actionLayer(x)
        
        #Q = (V + (A - tf.math.reduce_mean(A,axis=1,keepdims=True)))
        
        return action
    
        
    
    def get_config(self):
        return {"nActions":self.nActions,"inputShape":self.inputShape}
        
    '''
    def adtantage(self,state):
        x = self.inputLayer(state)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        A = self.A(x)
        return A
    '''






class ReplayBuffer():
    def __init__(self,maxSize,inputShape):
        self.memSize = maxSize
        self.memCntr = 0
        self.stateMemory = np.zeros((self.memSize,*inputShape),dtype=np.float32)
        self.newStateMemory = np.zeros((self.memSize,*inputShape),dtype=np.float32)
        
        self.actionMemory = np.zeros(self.memSize,dtype=np.int32)
        self.rewardMemory = np.zeros(self.memSize,dtype=np.int32)
        self.terminalMemory = np.zeros(self.memSize,dtype=np.bool)
        
        
    def storeTransition(self,state, action,reward,state_,done):
        index = self.memCntr % self.memSize
        #print(reward)
        #print(index)
        #print(state)
        self.stateMemory[index] = state
        self.newStateMemory[index] = state_
        self.rewardMemory[index] = reward
        self.actionMemory[index] = action
        self.terminalMemory[index] = done
        
        self.memCntr += 1
    
    def sampleBuffer(self, batchSize):
        maxMemory = min(self.memCntr,self.memSize)
        batch = np.random.choice(maxMemory,batchSize,replace=False)
        #print("Batch:",batch)
        states = self.stateMemory[batch]
        newStates = self.newStateMemory[batch]
        actions = self.actionMemory[batch]
        rewards = self.rewardMemory[batch]
        done = self.terminalMemory[batch]
        return states,actions,rewards,newStates,done
    
    
    
    
    
    
    
    
    
    
class Agent():
    def __init__(self,lr,gamma,nActions,epsilon,batchSize,inputDims,epsilonDec=1e-3,epsilonMin=0.01
                 ,memSize=10000, fname='dueling_dqn',replace=100,network=DeepQNetwork):
        self.actionSpace = np.array([i for i in range(nActions)])
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDec = epsilonDec
        self.epsilonMin = epsilonMin
        self.fname = fname
        self.replace = replace
        self.batchSize = batchSize
        
        self.learnStepCounter = 0
        self.memory = ReplayBuffer(memSize,inputDims)
        self.qEval = network(nActions,inputDims)
        self.qNext = network(nActions,inputDims)
        
        
        self.qEval.compile(optimizer=kOptimizers.Adam(learning_rate=lr),loss='mean_squared_error')
        self.qNext.compile(optimizer=kOptimizers.Adam(learning_rate=lr),loss='mean_squared_error')
        
    def storeTransition(self,state,action,reward,newState,done):
        self.memory.storeTransition(state,action,reward,newState,done)
        
    def chooseAction(self,observation):
        state = np.array([observation])
        networkActionQ = self.qEval(state)[0]
        networkAction = tf.math.argmax(networkActionQ).numpy()
        if(np.random.random() < self.epsilon):
            #chose random action that is not the same as networkAction
            action = np.random.choice(self.actionSpace[self.actionSpace != networkAction])
            
        else:
            state = np.array([observation])
            action = networkAction
            #actions = self.qEval(net)
            #print(actions)
            #action = tf.math.argmax(actions,axis=1).numpy()[0]
            
        return action,networkActionQ.numpy()[action]
    
    def learn(self):
        if(self.memory.memCntr < self.batchSize):
            return
        
        
        #if it has learned x amount of times, put main network's weights into the qNext network
        if(self.learnStepCounter % self.replace == 0):
            #print(self.qEval.get_weights())
            self.qNext.set_weights(self.qEval.get_weights())
        
        
        states,actions,rewards,states_, dones = self.memory.sampleBuffer(self.batchSize)
        

        
        
        

        #predicting future rewards
        qNext = self.qNext(states_)
        
        
        #getting the maximum reward from each state reward prediction
        qNext = tf.math.reduce_max(qNext, axis=1, keepdims=True).numpy()
        
        #predictiong q values for states before action
        qTarget = self.qEval(states).numpy()
        
        
        
        #itterates over each sample changing the q value for each action done
        # to be de possible best outcome of future actions
        for idx,terminal in enumerate(dones):
            #print(idx)
            if terminal:
                qNext[idx] = 0.0
            qTarget[idx][actions[idx]] = rewards[idx] + self.gamma*qNext[idx]
        
        
        self.qEval.train_on_batch(states,qTarget)
        
        self.epsilon = self.epsilon - self.epsilonDec if self.epsilon > self.epsilonMin else self.epsilonMin
        
        self.learnStepCounter += 1
        
    def saveModel(self):
        self.qEval.save_weights(self.fname,save_format="tf")
        
    def prepNetworksForLoad(self,observation):
        observation = np.array([observation])
        self.qEval(observation)
        self.qNext(observation)
        
    def loadModel(self):
        #print("Before loading: \n")
        #print(self.qEval.get_weights())
        self.qEval.load_weights(self.fname)
        #print("\n\n\n After loading: \n")
        #print(self.qEval.get_weights())
        #self.qNext.set_weights(self.qEval.get_weights())
        
        
        
    def changeEpsMin(self,value):
        self.epsilonMin = value
        
    
   
    
    
    
    
    
    
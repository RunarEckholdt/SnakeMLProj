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


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


#tf.config.gpu_options.allow_growth = True


class DeepQNetwork(keras.Model):
    def __init__(self,nActions,inputShape,fc1Dims,fc2Dims,fc3Dims):
        super(DeepQNetwork,self).__init__()
        self.inputLayer = keras.layers.Flatten(input_shape=inputShape,dtype=np.int32)
        self.denseLayer1 = keras.layers.Dense(fc1Dims,activation='sigmoid')
        self.denseLayer2 = keras.layers.Dense(fc2Dims,activation='sigmoid')
        self.denseLayer3 = keras.layers.Dense(fc3Dims,activation='sigmoid')
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
        self.stateMemory = np.zeros((self.memSize,*inputShape),dtype=np.int32)
        self.newStateMemory = np.zeros((self.memSize,*inputShape),dtype=np.int32)
        
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
                 ,memSize=10000, fname='dueling_dqn',fc1Dims=256,fc2Dims=256,fc3Dims=128,replace=100):
        self.actionSpace = [i for i in range(nActions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDec = epsilonDec
        self.epsilonMin = epsilonMin
        self.fname = fname
        self.replace = replace
        self.batchSize = batchSize
        
        self.learnStepCounter = 0
        self.memory = ReplayBuffer(memSize,inputDims)
        self.qEval = DeepQNetwork(nActions,inputDims,fc1Dims,fc2Dims,fc3Dims)
        self.qNext = DeepQNetwork(nActions,inputDims,fc1Dims,fc2Dims,fc3Dims)
        
        
        self.qEval.compile(optimizer=kOptimizers.Adam(learning_rate=lr),loss='mean_squared_error')
        self.qNext.compile(optimizer=kOptimizers.Adam(learning_rate=lr),loss='mean_squared_error')
        
    def storeTransition(self,state,action,reward,newState,done):
        self.memory.storeTransition(state,action,reward,newState,done)
        
    def chooseAction(self,observation):
        state = np.array([observation])
        networkAction = self.qEval(state)
        if(np.random.random() < self.epsilon):
            while True:
                action = np.random.choice(self.actionSpace)
                if action != networkAction:
                    break
        else:
            state = np.array([observation])
            actions = self.qEval(state)
            #print(actions)
            action = tf.math.argmax(actions,axis=1).numpy()[0]
            
        return action
    
    def learn(self):
        if(self.memory.memCntr < self.batchSize):
            return
        
        
        #if it has learned x amount of times, put main network's weights into the qNext network
        if(self.learnStepCounter % self.replace == 0):
            self.qNext.set_weights(self.qEval.get_weights())
        
        
        states,actions,rewards,states_, dones = self.memory.sampleBuffer(self.batchSize)
        
        #predicting the q values for states_
        
        
        
        #time.sleep(5)
        #predicting future rewards
        qNext = self.qNext(states_)
        #print(qNext)
        #print()
        #time.sleep(5)
        
        #getting the maximum reward from each state reward prediction
        qNext = tf.math.reduce_max(qNext, axis=1, keepdims=True).numpy()
        
        
        qTarget = self.qEval(states).numpy()
        #print(dones)
        #print(actions)
        #print(rewards)
        #print(self.gamma)
        for idx,terminal in enumerate(dones):
            #print(idx)
            if terminal:
                qNext[idx] = 0.0
            qTarget[idx][actions[idx]] = rewards[idx] + self.gamma*qNext[idx]
        
        #print(states)
        #print(qNext)
        #time.sleep(5)
        #print()
        #print(qTarget)
        #time.sleep(5)
        self.qEval.train_on_batch(states,qTarget)
        #time.sleep(2)
        self.epsilon = self.epsilon - self.epsilonDec if self.epsilon > self.epsilonMin else self.epsilonMin
        
        self.learnStepCounter += 1
        
    def saveModel(self):
        self.qEval.save_weights(self.fname,save_format="tf")
        
        
    def loadModel(self):
        self.qEval.load_weights(self.fname)
        self.qNext.set_weights(self.qEval.get_weights())
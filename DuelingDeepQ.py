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





class DuelingDeepQNetwork(keras.Model):
    def __init__(self,nActions,fc1Dims,fc2Dims):
        super(DuelingDeepQNetwork,self).__init__()
        self.inputLayer = keras.layers.Flatten(input_shape=(10,10),dtype=np.int32)
        self.dense1 = keras.layers.Dense(fc1Dims,activation='relu')
        self.dense2 = keras.layers.Dense(fc2Dims,activation='relu')
        self.V = keras.layers.Dense(1,activation=None)
        self.A = keras.layers.Dense(nActions,activation=None)
    
    def call(self,state):
        x = self.inputLayer(state)
        x = self.dense1(x)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        
        Q = (V + (A - tf.math.reduce_mean(A,axis=1,keepdims=True)))
        
        return Q
        
    
    def adtantage(self,state):
        x = self.inputLayer(state)
        x = self.dense1(x)
        x = self.dense2(x)
        A = self.A(x)
        return A






class ReplayBuffer():
    def __init__(self,maxSize,inputShape):
        self.memSize = maxSize
        self.memCntr = 0
        self.stateMemory = np.zeros((self.memSize, *inputShape),dtype=np.int32)
        self.newStateMemory = np.zeros((self.memSize,*inputShape),dtype=np.int32)
        
        self.actionMemory = np.zeros(self.memSize,dtype=np.int32)
        self.rewardMemory = np.zeros(self.memSize,dtype=np.int32)
        self.terminalMemory = np.zeros(self.memSize,dtype=np.bool)
        
        
    def storeTransition(self,state, action,reward,state_,done):
        index = self.memCntr % self.memSize
        #print(reward)
        #print(index)
        self.stateMemory[index] = state
        self.newStateMemory[index] = state_
        self.rewardMemory[index] = reward
        self.actionMemory[index] = action
        self.terminalMemory[index] = done
        
        self.memCntr += 1
    
    def sampleBuffer(self, batchSize):
        maxMemory = min(self.memCntr,self.memSize)
        batch = np.random.choice(maxMemory,batchSize,replace=False)
        states = self.stateMemory[batch]
        newStates = self.newStateMemory[batch]
        actions = self.actionMemory[batch]
        rewards = self.rewardMemory[batch]
        done = self.terminalMemory[batch]
        return states,actions,rewards,newStates,done
    
    
    
    
    
    
    
    
    
    
class Agent():
    def __init__(self,lr,gamma,nActions,epsilon,batchSize,inputDims,epsilonDec=1e-3,epsilonMin=0.01
                 ,memSize=100000, fname='dueling_dqn.keras',fc1Dims=128,fc2Dims=128,replace=100):
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
        self.qEval = DuelingDeepQNetwork(nActions,fc1Dims,fc2Dims)
        self.qNext = DuelingDeepQNetwork(nActions,fc1Dims,fc2Dims)
        
        self.qEval.compile(optimizer=kOptimizers.Adam(learning_rate=lr),loss='mean_squared_error')
        
        self.qNext.compile(optimizer=kOptimizers.Adam(learning_rate=lr),loss='mean_squared_error')
        
    def storeTransition(self,state,action,reward,newState,done):
        self.memory.storeTransition(state,action,reward,newState,done)
        
    def chooseAction(self,observation):
        if(np.random.random() < self.epsilon):
            action = np.random.choice(self.actionSpace)
        else:
            state = np.array([observation])
            actions = self.qEval.adtantage(state)
            #print(actions)
            action = tf.math.argmax(actions,axis=1).numpy()[0]
            
        return action
    
    def learn(self):
        if(self.memory.memCntr < self.batchSize):
            return
        
        if(self.learnStepCounter % self.replace == 0):
            self.qNext.set_weights(self.qEval.get_weights())
        
        states,actions,rewards,states_, dones = self.memory.sampleBuffer(self.batchSize)
        
        qPred = self.qEval(states)
        qNext = tf.math.reduce_max(self.qNext(states_), axis=1, keepdims=True).numpy()
        qTarget = np.copy(qPred)
        #print(dones)
        #print(actions)
        #print(rewards)
        #print(self.gamma)
        for idx,terminal in enumerate(dones):
            #print(idx)
            if terminal:
                qNext[idx] = 0.0
            qTarget[idx,actions[idx]] = rewards[idx] + self.gamma*qNext[idx]
        
        #print(states)
        #print(qTarget)
        self.qEval.train_on_batch(states,qTarget)
        #time.sleep(2)
        self.epsilon = self.epsilon - self.epsilonDec if self.epsilon > self.epsilonMin else self.epsilonMin
        
        self.learnStepCounter += 1
        
    def saveModel(self):
        self.qEval.save_weights(self.fname,overwrite=True,save_format="tf")
        
    def loadModel(self):
        self.qEval = kModel.load_weights(self.fname)
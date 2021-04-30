# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:45:37 2021

@author: Runar
"""

from SnakeGame import SnakeGame
#from DeepQAgent import Agent
from tensorflow import keras
import tensorflow as tf
from DeepQAgent import DeepQNetworkConv
import time
import numpy as np

choice = 1

game = SnakeGame(10,False)

#network = DeepQNetworkConv()

conv1 = keras.layers.Convolution1D(16, 8, strides=1,activation='relu',input_shape=(11,10),dtype=tf.float32)



obs = np.array([game.getObservation()],dtype=np.float32)

output= conv1(obs)

print(output)





actionSpace = np.array([i for i in range(3)])
print(actionSpace)
print(actionSpace[actionSpace != choice])
#action = np.random.choice(actionSpace)



input()
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:45:37 2021

@author: Runar
"""
import tensorflow as tf
from tensorflow import keras
from DuelingDeepQ import DeepQNetwork
import time
import numpy as np
#wow = tf.constant([[2,5,4,7,6,3]
 #                 ,[4,8,7,2,1,5]
  #                ,2])
 
#arr = np.array([[2,5,4,7,6,3]
 #         ,[4,8,7,2,1,5]
  #        ,2])
#print(arr.shape)

l = 20

shape = (10,10)

additional = np.zeros(10)


heck = np.zeros((10,10))

new = np.append(heck,[additional],axis=0)


print(new)
print(new.shape)

#new = np.append(heck,additional)
#print(new)



#dQN = DeepQNetwork(2,(2,6),20,10)



#flat = dQN.inputLayer(wow)

#print(flat.output_shape)


#model = keras.Sequential()
#model.add(keras.layers.Flatten(input_shape=(3,)))
#flat = keras.layers.Flatten(input_shape=(2,6))
#print(model.output_shape)

#heck = flat(wow).numpy()

#print(heck)


#print( wow[1][2])
#print( tf.math.reduce_max(wow,axis=1,keepdims=True).numpy())



#epsilon = (epsilon > epsilonMin)? epsilon - epsilonDec : epsilonMin;

input()
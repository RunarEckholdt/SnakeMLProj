# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:45:37 2021

@author: Runar
"""
import tensorflow as tf

wow = tf.constant([[2,5,4,7,6,3]
                  ,[4,8,7,2,1,5]])


print( wow[1][2])
#print( tf.math.reduce_max(wow,axis=1,keepdims=True).numpy())
input()
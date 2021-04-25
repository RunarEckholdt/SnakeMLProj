# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:18:49 2021

@author: Runar
"""

from snakeMap import SnakeGame
from DuelingDeepQ import Agent
import time


'''    
game = SnakeGame()
gameExited = False
gameGameover = False
while (not gameExited and not gameGameover):
    gameExited = game.updateDirection()
    gameGameover = game.checkForTick()
    
if(gameGameover):
    print("Game Over!")
    input()
'''

amountOfGames = 1000
snakeGameSize = 10
game = SnakeGame(snakeGameSize,False)
loadModel = False



agent = Agent(gamma=0.99,epsilon=1,lr=1e-3,inputDims=[10,10],epsilonDec=1e-3,memSize=100000,
              batchSize=64,epsilonMin=0.01,fc1Dims=128,fc2Dims=128,replace=100,nActions=4)

if(loadModel):
    agent.loadModel()

for i in range(amountOfGames):
    print("Game:",i)
    done = False
    observation = game.getObservation()
    while not done:
        action = agent.chooseAction(observation)
        #print("Action=",action)
        reward, observation_, done = game.step(action)
        agent.storeTransition(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()
        #time.sleep(0.1)
    score = game.getScore()
    print("Game %i ended with score %i" %(i,score))
    game.reset()
agent.saveModel()
input()
        
        


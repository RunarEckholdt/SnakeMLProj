# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:18:49 2021

@author: Runar
"""

from snakeMap import SnakeGame
from DuelingDeepQ import Agent
import time
import keyboard
from os import system





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

amountOfGames = 2000
snakeGameSize = 10
game = SnakeGame(snakeGameSize,False)
loadModel = True



agent = Agent(gamma=0.99,epsilon=1,lr=1e-3,inputDims=[10,10],epsilonDec=1e-3,memSize=100000,
              batchSize=64,epsilonMin=0.01,fc1Dims=128,fc2Dims=128,replace=100,nActions=4)

if(loadModel):
    agent.loadModel()

#percentage = 0

for i in range(amountOfGames):
    #print("Game:",i)
    done = False
    observation = game.getObservation()
    lastAction = 0
    while not done:
        action = agent.chooseAction(observation)
        if(action + 2 == lastAction or action -2 == lastAction):
            reward = -10
            done = False
            agent.storeTransition(observation, action, reward, observation, done)
            agent.learn()
        else:
            #print("Action=",action)
            reward, observation_, done = game.step(action)
            agent.storeTransition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
            lastAction = action
        
        time.sleep(0.1)
    score = game.getScore()
    if(keyboard.is_pressed("Esc")):
        break
    #print("Game %i ended with score %i" %(i,score))
    '''
    if((i//amountOfGames)* 100 != percentage):
        percentage = i//amountOfGames * 100
        system("cls")
        print(str(i) + "%")
        '''
    game.reset()
#agent.saveModel()
print("Done!")
input()
        
        


# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:18:49 2021

@author: Runar
"""

from SnakeGame import SnakeGame
from DeepQAgent import Agent
from DeepQAgent import DeepQNetworkConv
import time
import keyboard
from os import system
import matplotlib.pyplot as plt






amountOfGames = 40000
snakeGameSize = 10
game = SnakeGame(snakeGameSize,False)
loadModel = True
saveAnimation = True



agent = Agent(gamma=0.91,
              epsilon=0.1,
              lr=1e-3,
              inputDims=(11,10),
              epsilonDec=1e-6,
              memSize=10000,
              batchSize=64,
              epsilonMin=0.05,
              replace=100,
              nActions=3,
              network=DeepQNetworkConv)



obs = game.getObservation()
agent.prepNetworksForLoad(obs)
if(loadModel):
    agent.loadModel()



doPrint = False
printQValues = False
printEps = False

scores = []
fruits = []
matchNumbers = []
qValues = []


for i in range(amountOfGames):

    '''
    if(i == 3000):
        print("Changed epsilonMin to",0.2)
        agent.changeEpsMin(0.2)
    elif(i == 5000):
        print("Changed epsilonMin to",0.1)
        agent.changeEpsMin(0.1)
    '''
    # if(i == 10000):
    #     print("Changed epsilonMin to",0.05)
    #     agent.changeEpsMin(0.05)
    # elif(i == 10000):
    #     print("Changed epsilonMin to 0.01")
    #     agent.changeEpsMin(0.01)
    
    done = False
    observation = game.getObservation()
    
    
    while not done:
        
        if(keyboard.is_pressed('p')):
            doPrint = not doPrint
            time.sleep(0.5)
        if(keyboard.is_pressed('e')):
            printEps= not printEps
            time.sleep(0.5)
        if(keyboard.is_pressed('q')):
            printQValues = not printQValues
            time.sleep(0.5)
        
        
        while True:
            action,Q,wasEpsilon,qList = agent.chooseAction(observation,
                                                     returnQ=True,
                                                     returnIfEpsilon=True,
                                                     returnQList=True,
                                                     printQValues=printQValues)
            if(not wasEpsilon):
                break
            elif(not game.snakeMap.predictDeathByAction(action)):
                break
            
        
        qValues.append(Q)
        game.inputQValues(qList)
        reward, observation_, done = game.step(action,doPrint)
        agent.storeTransition(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()
       
        if doPrint:
            time.sleep(0.1)
    score = game.getScore()
    if(keyboard.is_pressed("Esc")):
        break
    
    scores.append(score)
    fruits.append(game.getFruitsEaten())
    matchNumbers.append(i)
    printStr = f"Game {i:>5} ended with score: {score:>10}"
    
    
    if(printEps):
        eps = agent.epsilon
        printStr += f" Epsilon{eps:>3.2f}"
    
    print(printStr)
   
    game.reset()


print("Done!")

fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()
fig3,ax3 = plt.subplots()
ax1.scatter(matchNumbers,scores,color='b',marker='o',alpha=0.5)
ax1.set_title("Scores")
ax2.scatter(matchNumbers, fruits,color='g',marker='o',alpha=0.5)
ax2.set_title("Fruits")
qCount=[i for i in range(len(qValues))]
ax3.scatter(qCount,qValues,color='r',marker='o',alpha=0.1)
ax3.set_title("Q Values for decision")



plt.show()


yPressed = False
nPressed = False

print("Save model? (y/n): ")
while(not yPressed and not nPressed):
    if(keyboard.is_pressed('y')):
        yPressed = True
        agent.saveModel()
        print("Model is saved")
    elif(keyboard.is_pressed('n')):
        nPressed = True
        print("Model was not saved")
input("Press enter to see replay of best game:")
game.bestReplay.animate(saveAnimation=saveAnimation)



    
        
    

        
        


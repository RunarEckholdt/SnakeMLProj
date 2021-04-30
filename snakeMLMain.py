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

amountOfGames = 40000
snakeGameSize = 10
game = SnakeGame(snakeGameSize,False)
loadModel = True

#memSize 5000 gamma .92 epsilonMin .07 no sucsess


agent = Agent(gamma=0.93,epsilon=0.4,lr=1e-3,inputDims=(11,10),epsilonDec=1e-4,memSize=10000,
              batchSize=64,epsilonMin=0.1,replace=100,nActions=3,network=DeepQNetworkConv)

obs = game.getObservation()
agent.prepNetworksForLoad(obs)
if(loadModel):
    agent.loadModel()

#percentage = 0

doPrint = False

scores = []
fruits = []
matchNumbers = []
qValues = []


for i in range(amountOfGames):
    #print("Game:",i)
    '''
    if(i == 3000):
        print("Changed epsilonMin to",0.2)
        agent.changeEpsMin(0.2)
    elif(i == 5000):
        print("Changed epsilonMin to",0.1)
        agent.changeEpsMin(0.1)
    '''
    if(i == 10000):
        print("Changed epsilonMin to",0.05)
        agent.changeEpsMin(0.05)
    # elif(i == 10000):
    #     print("Changed epsilonMin to 0.01")
    #     agent.changeEpsMin(0.01)
    
    done = False
    observation = game.getObservation()
    #lastAction = 0
    #direction = 0
    '''
    if i > 1000 and i < 1100:
        doPrint = True
    elif i > 2000 and i < 2100:
        doPrint = True
    else:
        doPrint = False
    '''
    while not done:
        if(keyboard.is_pressed('p')):
            doPrint = True
        else:
            doPrint = False
        
        action,Q = agent.chooseAction(observation,True)
        qValues.append(Q)
        reward, observation_, done = game.step(action,doPrint)
        agent.storeTransition(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()
        #lastAction = action
        if doPrint:
            time.sleep(0.1)
    score = game.getScore()
    if(keyboard.is_pressed("Esc")):
        break
    
    scores.append(score)
    fruits.append(game.getFruitsEaten())
    matchNumbers.append(i)
    print("Game %i ended with score %i" %(i,score))
    '''
    if((i//amountOfGames)* 100 != percentage):
        percentage = i//amountOfGames * 100
        system("cls")
        print(str(i) + "%")
        '''
    game.reset()


print("Done!")
#plt.plot(matchNumbers,scores,'bo')
fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()
fig3,ax3 = plt.subplots()
ax1.plot(matchNumbers,scores,'bo')
ax1.set_title("Scores")
ax2.plot(matchNumbers, fruits,'go')
ax2.set_title("Fruits")
ax3.plot(qValues)
ax3.set_title("Q Values for decision")


#plt.plot(matchNumbers,fruits,'go')
plt.show()

#inp = input("Save model (y/n): ").lower()
yPressed = False
nPressed = False

print("Save model? (y/n): ")
while(not yPressed and not nPressed):
    if(keyboard.is_pressed('y')):
        yPressed = True
        agent.saveModel()
    elif(keyboard.is_pressed('n')):
        nPressed = True


    
        
    

        
        


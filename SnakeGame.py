# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:19:43 2021

@author: Runar
"""

import numpy as np
import random
import keyboard
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import system
        
        

directionDict = {"up":0,"right":1,"down":2,"left":3}
vectorToDirectionDict = {(-1,0):0,(0,1):1,(1,0):2,(0,-1):3}

mapValues = {"empty":0,"wall":-4,"head":1,"body":-1,"fruit":4} 

mapValuesP = {mapValues["empty"]:" ",
               mapValues["wall"]: "W",
               mapValues["head"]: "H",
               mapValues["body"]: "B",
               mapValues["fruit"]:"F"}
 



snakeMaxLife = 70

class Fruit:
    def __init__(self,mapSize):
        self.__mapSize = mapSize
        self.newPosition()
    
    def newPosition(self):
        self.__position = Coordinate(random.randint(1,self.__mapSize-2),random.randint(1,self.__mapSize-2))
        return self.__position
    
    def getPosition(self):
        return self.__position
    
    

class Coordinate:
    def __init__(self,y,x):
        self.__y = y
        self.__x = x
    
    def getX(self):
        return self.__x
    
    
    def getY(self):
        return self.__y
    
    
    def __add__(self,movement):
        if(type(movement) != list and type(movement) != tuple):
            raise Exception("Addition on coordinate must be used on a tuple or list")
        return Coordinate(self.__y + movement[0], self.__x + movement[1])
    
    def __str__(self):
        return "Y: "+ str(self.__y) + " X: " + str(self.__x)
    
    def __eq__(self,coord):
        return self.__y == coord.getY() and self.__x == coord.getX()
    
    
    
    
        

class Snake:
    def __init__(self,startCoordinate,startDirection):
        
        if(type(startCoordinate)==Coordinate):
            self.__headPosition = startCoordinate
        elif(type(startCoordinate)==tuple or type(startCoordinate)==list):
            self.__headPosition = Coordinate(startCoordinate[0],startCoordinate[1])
        else:
            sCType = str(type(startCoordinate))
            raise Exception("startCoordinate was of type" + sCType)
        self.__bodyPositions = []
        
        if(startDirection == directionDict["up"]):
            self.__bodyPositions.append(self.__headPosition + [1,0])
        elif(startDirection == directionDict["right"]):
            self.__bodyPositions.append(self.__headPosition + [0,-1])
        elif(startDirection == directionDict["down"]):
            self.__bodyPositions.append(self.__headPosition + [-1,0])
        elif(startDirection == directionDict["left"]):
            self.__bodyPosition.append(self.__headPosition + [0,1])
        
        self.__direction = startDirection
        
        self.life = snakeMaxLife
        
    def move(self,movement):
        if(type(movement) != list and type(movement) != tuple):
            raise Exception("movement parameter must be a tuple or list")
        lastBodyPosition = self.__headPosition
        self.__headPosition = self.__headPosition + movement
       
        for i in range(len(self.__bodyPositions)):
            self.__bodyPositions[i], lastBodyPosition = lastBodyPosition , self.__bodyPositions[i]
            
        self.__behindTailPosition = lastBodyPosition
        self.__direction = vectorToDirectionDict[(self.__headPosition.getY()-self.__bodyPositions[0].getY(),
                                                  self.__headPosition.getX()-self.__bodyPositions[0].getX())]
        return self.__headPosition
        
    
    def eatFruit(self):
        self.__bodyPositions.append(self.__behindTailPosition)
        self.life = snakeMaxLife
        
     
    def getDirection(self):
        return self.__direction
    
    def fetchPositions(self):
        yield self.__headPosition,"h"
        for bodyPosition in self.__bodyPositions:
            yield bodyPosition,"b"
            
    def getHeadPosition(self):
        return self.__headPosition
            
    def getOldTailPosition(self):
        return self.__behindTailPosition
        
          
class GameReplay:
    def __init__(self,startScene):
        self.gameScenes = [startScene.copy()]
        self.score = 0
        
        
       
    def addGameScene(self,scene):
        if(type(scene) != np.ndarray):
            raise Exception("Scene must be a numpy array")
        self.gameScenes.append(scene.copy())
    def setScore(self, score):
        self.score = score
    def replay(self):
        for gameScene in self.gameScenes:
            mapStr = ""
            for layer in gameScene:
                for value in layer:
                    mapStr += mapValuesP[value] + " "
                mapStr += "\n"
            system('cls')
            print(mapStr)
            time.sleep(1)  
    def animate(self,saveAnimation=False):
        
        self.fig = plt.figure()
        
        
        frames = np.arange(0,len(self.gameScenes),1)
        a = animation.FuncAnimation(self.fig,self.__update,frames=frames,interval=500)
        plt.show()
        if saveAnimation:
            gifName = input("Enter name for animation file:")
            writerGif = animation.PillowWriter(fps=2)
            a.save(gifName+".gif",writer=writerGif)
        return a
    def __update(self,frameNumber):
        self.fig.clf()
        ax = self.fig.gca()
        
        for y,layer in enumerate(self.gameScenes[frameNumber]):
            for x,value in enumerate(layer):
                if value == mapValues["wall"]:
                    ax.plot(y,x,color='#000000',marker='o',markersize=10)
                elif value == mapValues["body"]:
                    ax.plot(y,x,color='#00FF00',marker='o',markersize=4)
                elif value == mapValues["fruit"]:
                    ax.plot(y,x,color='#FF0000',marker='o',markersize=6)
                elif value == mapValues["head"]:
                    ax.plot(y,x,color='#00FF00',marker='o',markersize=6)
        ax.axis('off')
                
        
        

class SnakeMap:
    def __init__(self,size,mapForUser):
        
        self.fruitsEaten = 0
        
        self.__map = np.full(shape=(size,size),fill_value=mapValues["empty"],dtype=np.int32)
        
        self.snake = Snake((size-3,size-3),directionDict["up"])
        self.__fruit = Fruit(size)
        self.__size = size
        self.__updateSnake()
        self.__updateFruit()
        self.__setBorderWalls()
        self.__gameover = False

        
        
    
    def doTick(self,action):
        movement = self.__calculateMovement(action)   
        newPosition = self.snake.move(movement)
        additionalScore = 0
        
        if(self.atPos(newPosition) == mapValues["wall"] or self.atPos(newPosition) == mapValues["body"]):
            self.__gameover = True
            additionalScore = -1000
            
        else:
            self.__changeMap(self.snake.getOldTailPosition(),mapValues["empty"])
            
        self.__updateSnake()
       
        if(newPosition == self.__fruit.getPosition()):
            additionalScore = 1000

            self.fruitsEaten += 1
            self.snake.eatFruit()
            while True:
                if(self.atPos(self.__fruit.newPosition()) == mapValues["empty"]):
                    self.__updateFruit()
                    break
        elif(not self.__gameover):
            self.snake.life -= 1
            if self.snake.life <= 0:
                self.__gameover = True
                additionalScore = -1000
            else:
                additionalScore = -1
        self.__updateFruit()
        return additionalScore,self.snake.getDirection()
        
            
    def getMap(self):
        return self.__map.copy()
            
    def isGameOver(self):
        return self.__gameover
    
    def __updateSnake(self):
        for position,part in self.snake.fetchPositions():
            if(part == "h"):
                p = mapValues["head"]
            elif(part=="b"):
                p = mapValues["body"]
            self.__changeMap(position,p)
            
    def __updateFruit(self):
        self.__changeMap(self.__fruit.getPosition(), mapValues["fruit"])
    
    def __changeMap(self,position, value):
        self.__map[position.getY()][position.getX()] = value
                
    
        
    def __str__(self):
        mapStr = ""
        
        for layer in self.__map:
            for value in layer:
                mapStr += mapValuesP[value] + " "
            mapStr += "\n"
        
        return mapStr
    

    def atPos(self,position):
        return self.__map[position.getY()][position.getX()]
    
    def __setBorderWalls(self):
        self.__map[0,:]=mapValues["wall"]
        self.__map[self.__size - 1,:] = mapValues["wall"]
        self.__map[1:self.__size-1,0] = mapValues["wall"]
        self.__map[1:self.__size-1,self.__size-1] = mapValues["wall"]
     
    def __calculateMovement(self,action):
        snakeDirection = self.snake.getDirection()
        #if left
        if(action == 0):
            newDirection = (snakeDirection + 3) % 4
        #if right
        elif(action == 2):
            newDirection = (snakeDirection + 1) % 4
        #forward
        else:
            newDirection = snakeDirection
        
        if(newDirection  == 0):
            movement = (-1,0)
        elif(newDirection  == 1):
            movement = (0,1)
        elif(newDirection  == 2):
            movement = (1,0) 
        else:
            movement = (0,-1)
        return movement
    #returns True if it will die from action else False
    def predictDeathByAction(self,action):
        movement = self.__calculateMovement(action)
        
        snakeHeadPosition = self.snake.getHeadPosition()
        
        newPosition = snakeHeadPosition + movement
        
        return (self.atPos(newPosition) == mapValues["wall"] or self.atPos(newPosition) == mapValues["body"])
 
    
  
    
  
    
  
    
  
    
  
    
class SnakeGame:
    def __init__(self,size,userPlayed):
        self.__userPlayed = userPlayed
        self.snakeMap = SnakeMap(size,userPlayed)
        self.__directions={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
        self.__direction = self.__directions["up"]
        self.__lastTick = 0
        self.__score = 0
        self.__gameSize = size
        self.__snakeDirection = 0
        self.__currentReplay = GameReplay(self.snakeMap.getMap())
        self.bestReplay = GameReplay(self.snakeMap.getMap())
        
        
    
    def reset(self):
        if(self.bestReplay.score < self.__score):
            self.__currentReplay.setScore(self.__score)
            self.bestReplay = self.__currentReplay
        self.snakeMap = SnakeMap(self.__gameSize,self.__userPlayed)
        self.__currentReplay = GameReplay(self.snakeMap.getMap())
        self.__direction = self.__directions["up"]
        self.__lastTick = 0
        self.__score = 0
    
    def updateDirection(self):
        if(keyboard.is_pressed('w')):
            self.__direction=self.__directions["up"]
        elif(keyboard.is_pressed('a')):
            self.__direction=self.__directions["left"]
        elif(keyboard.is_pressed('s')):
            self.__direction=self.__directions["down"]
        elif(keyboard.is_pressed('d')):
            self.__direction=self.__directions["right"]
        return keyboard.is_pressed("Esc")
        
    def checkForTick(self):
        if(time.time() - self.__lastTick > 1.0):
            self.__lastTick = time.time()
            self.__score += self.snakeMap.doTick(self.__direction)
            system('cls')
            print("Score:",self.__score)
            print(self.snakeMap)
            return self.snakeMap.isGameOver()
        
    
    def step(self,action,doPrint):
        
        
        reward,self.__snakeDirection = self.snakeMap.doTick(action)

        done = self.snakeMap.isGameOver()
        
        newObservation = self.getObservation()
        
        self.__score += reward
        if doPrint:
            system('cls')
            print("Score:",self.__score)
            print(self.snakeMap)
        self.__currentReplay.addGameScene(self.snakeMap.getMap())
        return reward,newObservation,done
    
    
    def getObservation(self):
        directionObservation = np.zeros(self.__gameSize)
        directionObservation[self.__snakeDirection] = 1
        directionObservation[4] = 1 - (self.snakeMap.snake.life / snakeMaxLife)
        obs = np.append(self.snakeMap.getMap(),[directionObservation],axis=0)
        obs = np.array(obs,dtype=np.float32)
        return obs
    
    def getScore(self):
        return self.__score
    
    def getFruitsEaten(self):
        return self.snakeMap.fruitsEaten    
    
    def printMap(self):
        print(self.snakeMap)
            
            
        
        
    
    

    
        
        
        
    
    
    
    
    
    
    
    
    
    
        
        


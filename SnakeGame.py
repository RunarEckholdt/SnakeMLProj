# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:19:43 2021

@author: Runar
"""

import numpy as np
import random
import keyboard
import time

from os import system
        
        

directionDict = {"up":0,"right":1,"down":2,"left":3}
vectorToDirectionDict = {(-1,0):0,(0,1):1,(1,0):2,(0,-1):3}

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
        #print("Snake Constructor called")
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
        #self.__bodyPositions = [Coordinate(self.__headPosition.getY() +1,self.__headPosition.getX())]
        self.__direction = startDirection
        
        self.life = 70
        
    def move(self,movement):
        if(type(movement) != list and type(movement) != tuple):
            raise Exception("movement parameter must be a tuple or list")
        lastBodyPosition = self.__headPosition
        self.__headPosition = self.__headPosition + movement
       
        for i in range(len(self.__bodyPositions)):
            self.__bodyPositions[i], lastBodyPosition = lastBodyPosition , self.__bodyPositions[i]
            #print(lastBodyPosition)
        self.__behindTailPosition = lastBodyPosition
        self.__direction = vectorToDirectionDict[(self.__headPosition.getY()-self.__bodyPositions[0].getY(),
                                                  self.__headPosition.getX()-self.__bodyPositions[0].getX())]
        return self.__headPosition
        
    
    def eatFruit(self):
        self.__bodyPositions.append(self.__behindTailPosition)
        self.life = 70
        
     
    def getDirection(self):
        return self.__direction
    
    def fetchPositions(self):
        yield self.__headPosition,"h"
        for bodyPosition in self.__bodyPositions:
            yield bodyPosition,"b"
            
    def getOldTailPosition(self):
        return self.__behindTailPosition
        
          
        
        

class SnakeMap:
    def __init__(self,size,mapForUser):
        #print("Snake Map Constructor called")
        self.fruitsEaten = 0
        
        self.mapValues = {"empty":0,"wall":-4,"head":1,"body":-1,"fruit":4}
        self.__map = np.full(shape=(size,size),fill_value=0,dtype=np.int32)
        self.mapValuesP = {self.mapValues["empty"]:" ",
                           self.mapValues["wall"]:"w",
                           self.mapValues["head"]:"h",
                           self.mapValues["body"]:"b",
                           self.mapValues["fruit"]:"f"}
        
        self.__snake = Snake((size-3,size-3),directionDict["up"])
        self.__fruit = Fruit(size)
        self.__size = size
        self.__updateSnake()
        self.__updateFruit()
        self.__setBorderWalls()
        self.__gameover = False
        
        
    
    def doTick(self,action):
        snakeDirection = self.__snake.getDirection()
        
        if(action == 0):
            newDirection = (snakeDirection + 3) % 4
        elif(action == 2):
            newDirection = (snakeDirection + 1) % 4
        else:
            newDirection = snakeDirection
        
        if(newDirection  == 0):
            newPosition = self.__snake.move((-1,0))
        elif(newDirection  == 1):
            newPosition = self.__snake.move((0,1))
        elif(newDirection  == 2):
            newPosition = self.__snake.move((1,0))   
        else:
            newPosition = self.__snake.move((0,-1))     
        
        additionalScore = 0
        #print(newPosition)
        #print(self.__atPos(newPosition))
        #print(self.__map)
        if(self.__atPos(newPosition) == self.mapValues["wall"] or self.__atPos(newPosition) == self.mapValues["body"]):
            self.__gameover = True
            additionalScore = -1000
            
        else:
            self.__changeMap(self.__snake.getOldTailPosition(),self.mapValues["empty"])
            
        self.__updateSnake()
        #print(newPosition,self.__fruit.getPosition())
        if(newPosition == self.__fruit.getPosition()):
            additionalScore = 1000
            #print("Fruit eaten!")
            self.fruitsEaten += 1
            self.__snake.eatFruit()
            while True:
                if(self.__atPos(self.__fruit.newPosition()) == self.mapValues["empty"]):
                    self.__updateFruit()
                    break
        elif(not self.__gameover):
            self.__snake.life -= 1
            if self.__snake.life <= 0:
                self.__gameover = True
                additionalScore = -1000
            else:
                additionalScore = -10
        self.__updateFruit()
        return additionalScore,self.__snake.getDirection()
        
            
    def getMap(self):
        return self.__map.copy()
            
    def isGameOver(self):
        return self.__gameover
    
    def __updateSnake(self):
        for position,part in self.__snake.fetchPositions():
            if(part == "h"):
                p = self.mapValues["head"]
            elif(part=="b"):
                p = self.mapValues["body"]
            self.__changeMap(position,p)
            
    def __updateFruit(self):
        self.__changeMap(self.__fruit.getPosition(), self.mapValues["fruit"])
    
    def __changeMap(self,position, value):
        self.__map[position.getY()][position.getX()] = value
                
    
        
    def __str__(self):
        mapStr = ""
        
        for layer in self.__map:
            for value in layer:
                mapStr += self.mapValuesP[value] + " "
            mapStr += "\n"
        
        return mapStr
    

    def __atPos(self,position):
        return self.__map[position.getY()][position.getX()]
    
    def __setBorderWalls(self):
        self.__map[0,:]=self.mapValues["wall"]
        self.__map[self.__size - 1,:] = self.mapValues["wall"]
        self.__map[1:self.__size-1,0] = self.mapValues["wall"]
        self.__map[1:self.__size-1,self.__size-1] = self.mapValues["wall"]
        
        
    
    
class SnakeGame:
    def __init__(self,size,userPlayed):
        #print("Snake Game Constructor called")
        self.__userPlayed = userPlayed
        self.__snakeMap = SnakeMap(size,userPlayed)
        self.__directions={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
        self.__direction = self.__directions["up"]
        self.__lastTick = 0
        self.__score = 0
        self.__gameSize = size
        self.__snakeDirection = 0
        
        #print(self.__lastTick)
    
    def reset(self):
        self.__snakeMap = SnakeMap(self.__gameSize,self.__userPlayed)
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
            self.__score += self.__snakeMap.doTick(self.__direction)
            system('cls')
            print("Score:",self.__score)
            print(self.__snakeMap)
            return self.__snakeMap.isGameOver()
        
    
    def step(self,action,doPrint):
        
        
        reward,self.__snakeDirection = self.__snakeMap.doTick(action)
        directionObservation = np.zeros(self.__gameSize)
        directionObservation[self.__snakeDirection] = 1
        newObservation = np.append(self.__snakeMap.getMap(),[directionObservation],axis=0)
        done = self.__snakeMap.isGameOver()
        
        self.__score += reward
        if doPrint:
            system('cls')
            print("Score:",self.__score)
            print(self.__snakeMap)
        return reward,newObservation,done
    
    
    def getObservation(self):
        directionObservation = np.zeros(self.__gameSize)
        directionObservation[self.__snakeDirection] = 1
        return np.append(self.__snakeMap.getMap(),[directionObservation],axis=0)
    
    def getScore(self):
        return self.__score
    
    def getFruitsEaten(self):
        return self.__snakeMap.fruitsEaten    
            
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        


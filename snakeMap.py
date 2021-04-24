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
    def __init__(self,startCoordinate):
        print("Snake Constructor called")
        if(type(startCoordinate)==Coordinate):
            self.__headPosition = startCoordinate
        elif(type(startCoordinate)==tuple or type(startCoordinate)==list):
            self.__headPosition = Coordinate(startCoordinate[0],startCoordinate[1])
        else:
            sCType = str(type(startCoordinate))
            raise Exception("startCoordinate was of type" + sCType)
        self.__bodyPositions = [Coordinate(self.__headPosition.getY() -1,self.__headPosition.getX())]
       
        
    def move(self,movement):
        if(type(movement) != list and type(movement) != tuple):
            raise Exception("movement parameter must be a tuple or list")
        lastBodyPosition = self.__headPosition
        self.__headPosition = self.__headPosition + movement
       
        for i in range(len(self.__bodyPositions)):
            self.__bodyPositions[i], lastBodyPosition = lastBodyPosition , self.__bodyPositions[i]
            print(lastBodyPosition)
        self.__behindTailPosition = lastBodyPosition
        return self.__headPosition
        
    
    def eatFruit(self):
        self.__bodyPositions.append(self.__behindTailPosition)
        
     
    def fetchPositions(self):
        yield self.__headPosition,"h"
        for bodyPosition in self.__bodyPositions:
            yield bodyPosition,"b"
            
    def getOldTailPosition(self):
        return self.__behindTailPosition
        
          
        
        

class SnakeMap:
    def __init__(self,size):
        print("Snake Map Constructor called")
        self.__map = np.full(shape=(size,size),fill_value=" ",dtype=str)
        self.__snake = Snake((size-2,size-2))
        self.__fruit = Fruit(size)
        self.__size = size
        self.__updateSnake()
        self.__updateFruit()
    
    def doTick(self,movement):
        newPosition = self.__snake.move(movement)
        self.__updateSnake()
        print(newPosition,self.__fruit.getPosition())
        if(newPosition == self.__fruit.getPosition()):
            print("Fruit eaten!")
            self.__snake.eatFruit()
            while True:
                if(self.__atPos(self.__fruit.newPosition()) == " "):
                    self.__updateFruit()
                    break
        else:
            self.__changeMap(self.__snake.getOldTailPosition()," ")
            
            
    
    def __updateSnake(self):
        for position,part in self.__snake.fetchPositions():
            self.__changeMap(position,part)
            
    def __updateFruit(self):
        self.__changeMap(self.__fruit.getPosition(), "f")
    
    def __changeMap(self,position, value):
        self.__map[position.getY()][position.getX()] = value
                
    
        
    def __str__(self):
        return str(self.__map)
    

    def __atPos(self,position):
        return self.__map[position.getY()][position.getX()]
        
        
    
    
class SnakeGame:
    def __init__(self):
        print("Snake Game Constructor called")
        self.__snakeMap = SnakeMap(10)
        self.__directions={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
        self.__direction = self.__directions["up"]
        self.__lastTick = 0
        print(self.__lastTick)
    
    def updateDirection(self):
        if(keyboard.is_pressed('w')):
            self.__direction=self.__directions["up"]
        elif(keyboard.is_pressed('a')):
            self.__direction=self.__directions["left"]
        elif(keyboard.is_pressed('s')):
            self.__direction=self.__directions["down"]
        elif(keyboard.is_pressed('d')):
            self.__direction=self.__directions["right"]
    def checkForTick(self):
        if(time.time() - self.__lastTick > 1.0):
            self.__lastTick = time.time()
            self.__snakeMap.doTick(self.__direction)
            system('cls')
            print(self.__snakeMap)
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        


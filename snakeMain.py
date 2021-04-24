# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:18:49 2021

@author: Runar
"""

from snakeMap import SnakeGame
    
game = SnakeGame()

while True:
    game.updateDirection()
    game.checkForTick()
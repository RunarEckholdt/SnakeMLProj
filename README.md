# SnakeMLProj
Using machine learning to play snake

The project includes: 
  The retro game "snake".
    The game object includes a gamereplay class that can be saved as a gif.
  An agent using deep q reinforcement learning.
 
Parameters for the agent:
  lr : The learning rate for the optimizer
  gamma : The discount multiplier in the Q value calculation. Higher value will focus learning towards rewards further away in time. Lower value focuses on rewards close in time.
  nActions : Amount of possible decissions the agent can chose between.
  epsilon : The probability for the agent to do an action that is not the same as the action suggested from the network.
  batchSize : The amount of random replays the network will train on for each itteration. Larger batch will reduce the importance of each replay.
  inputDims : The input dimention for the neural network.
  epsilonDec : How much epsilon should decrese after each decision.
  epsilonMin : The minimum value for epsilon.
  memSize : Amount of replays saved in the replaybuffer.
  fname : Filename to save or load the model
  replace : Amount of training the network must do before using the newly trained network to predict the q values. (Assign weights of qEval into the qNext network)
  network : The neural network to use in the agent.
  
Packages needed for the project:
  Tensorflow
  Numpy
  Keyboard
  matplotlib
  random
  time
  os
  
  

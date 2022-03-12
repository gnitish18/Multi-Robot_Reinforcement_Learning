
import pygame, sys, time, random, os
from pygame.locals import *

import math
import numpy as np
import pickle

import tensorflow as tf
from objectClasses import *
from pong_episode_run_mod3 import *

# NOTE: may want to change rewards/metrics in this testing script, in that case redesign game_loop

white = [255, 255, 255]
black = [0, 0, 0]
clock = pygame.time.Clock()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Most args here are useless, just here as placeholders
    # Want to match these:
    parser.add_argument('--numPaddles', default = 2,type=int) # Total number of paddles in a team (e.g. if --numPaddles == 2, we have two on each team)
    parser.add_argument('--numBalls', default = 4,type=int) # Total number of balls in the game
    parser.add_argument('--paddleDist', default = 100.0,type=float) # Distance between paddles
    parser.add_argument('--padSpeed',default = 5.0,type=float) # Speed of paddles, original is 5
    parser.add_argument('--initBallSpeed',default = 2.0,type=float) # Speed of balls, original is 2
    parser.add_argument('--DQNint', default = 10,type=int) # How many episodes to wait between training DQN
    parser.add_argument('--stw', default = 10,type=int) # Score to win    
    parser.add_argument('--whichSide', default = 'False',type=str) # True is left, False is right -- refers to which side our trained agents would be playing on
    
    # These don't matter/aren't used in testing, or don't know
    parser.add_argument('--eps', default = 1.0,type=float) # Epsilon, initial percentage of exploratory behavior
    parser.add_argument('--epsDecay', default = 0.005,type=float) # Epsilon decay
    parser.add_argument('--memBufLen', default = 100000,type=int) # Max length of memory buffer
    parser.add_argument('--trainSetSize',default = 20000,type=int) # Subset of memory buffer that will be used in each set of training
    parser.add_argument('--gamma', default = .95,type=float) # Discount for TD loss
    parser.add_argument('--lr', default = .0000025,type=float) # Learning rate of DQN
    parser.add_argument('--lrDecay', default = .25,type=float) # Decay rate of DQN lr, per training*implement as per epoch?
    parser.add_argument('--epch',default=15,type=int)
    parser.add_argument('--batSize',default = 10,type=int)
    parser.add_argument('--huber', default = 'False',type=str)
   
    # Things that are definitely specific to training
    parser.add_argument('--noEps', default = 100,type=int) # Number of Episodes
    parser.add_argument('--train',default = 'False',type=str) # True if training... changes some saving/loading options
    parser.add_argument('--pretrain',default = 'True',type=str) # If using pretrained weights
    parser.add_argument('--yesRender', default = 'True',type=str)
    parser.add_argument('--withTFmodel', default = 'True',type=str)
    parser.add_argument('--indir',default = 'latest/',type=str) # If want to load weights from a specific subdirectory... defaults to the latest training (saved in trained_weights/latest)
    parser.add_argument('--savedir',default = 'latest/',type=str) # Specify directory to save selected stats in (will save everything, including weights, to trained_weights/<name>)
    
    args = parser.parse_args()

    print("INDIR",args.indir)
    print("INDIR",args.savedir)    
    pygame.init()
    init_game(args)

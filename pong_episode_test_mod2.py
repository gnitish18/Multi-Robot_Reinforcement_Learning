
import pygame, sys, time, random, os
from pygame.locals import *

import math
import numpy as np
import pickle

import tensorflow as tf
from objectClasses import *
from pong_episode_run_mod2 import *

# NOTE: may want to change rewards/metrics in this testing script, in that case redesign game_loop

white = [255, 255, 255]
black = [0, 0, 0]
clock = pygame.time.Clock()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eps', default = 1.0,type=float) # Epsilon, initial percentage of exploratory behavior
    parser.add_argument('--epsDecay', default = 0.005,type=float) # Epsilon decay
    parser.add_argument('--yesRender', default = True,type=bool) # We want to render for testing usually
    parser.add_argument('--withTFmodel', default = True,type=bool)
    parser.add_argument('--noEps', default = 100,type=int) # Number of Episodes
    parser.add_argument('--stw', default = 10,type=int) # Score to win
    parser.add_argument('--memBufLen', default = 50000,type=int) # Max length of memory buffer
    parser.add_argument('--gamma', default = .95,type=float) # Discount for TD loss
    parser.add_argument('--lr', default = .000005,type=float) # Learning rate of DQN
    parser.add_argument('--lrDecay', default = .25,type=float) # Decay rate of DQN lr, per training*implement as per epoch?
    parser.add_argument('--DQNint', default = 10,type=int) # How many episodes to wait between training DQN
    parser.add_argument('--epch',default=15,type=int)
    parser.add_argument('--batSize',default = 10,type=int)
    parser.add_argument('--trainSetSize',default = 10000,type=int)
    parser.add_argument('--TRAIN',default = False,type=bool) # TRAIN=False means Testing, and will save memory buffer
    parser.add_argument('--pretrain',default = True,type=bool) # If using pretrained weights
    parser.add_argument('--indir',default = 'latest/',type=str) # If want to load weights from a specific subdirectory... defaults to the latest training (saved in trained_weights/latest)
    parser.add_argument('--savedir',default = 'latest/',type=str) # Specify directory to save selected stats in (will save everything, including weights, to trained_weights/<name>)
    parser.add_argument('--padSpeed',default = 1.0,type=float) # Speed of paddles

    args = parser.parse_args()
    
    pygame.init()
    init_game(args)

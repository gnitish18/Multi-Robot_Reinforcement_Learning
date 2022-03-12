import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--numPaddles', default = 2,type=int) # Total number of paddles in a team (e.g. if --numPaddles == 2, we have two on each team)
parser.add_argument('--numBalls', default = 4,type=int) # Total number of balls in the game
parser.add_argument('--indir',default = '',type=str) # If want to load weights from a specific subdirectory... defaults to the latest training (saved in trained_weights/latest)
parser.add_argument('--savedir',default = 'Heuristic-Aided RL',type=str) # Specify directory to save selected stats in (will save everything, including weights, to trained_weights/<name>)
parser.add_argument('--DQNint', default = 10,type=int) # How many episodes to wait between training DQN

args = parser.parse_args()

# Process Inputs
# NOTE: uncomment if you aren't Alec
# indir = './trained_weights/'+args.indir
# savedir = './trained_weights/'+args.savedir
indir = args.indir
indir = args.savedir
title = savedir # A nice, short title
# indir = 'C:/Users/Alec/Documents/GitHub/Multi-Robot_Reinforcement_Learning/trained_weights/'+indir
indir = 'C:/Users/Alec/Documents/GitHub/Multi-Robot_Reinforcement_Learning/trained_weights/'
savedir = 'C:/Users/Alec/Documents/GitHub/Multi-Robot_Reinforcement_Learning/trained_weights/'+savedir

noPaddles = args.numPaddles
noBalls = args.numBalls
DQNint = args.DQNint

#indir = 'C:\Users\Alec\Documents\GitHub\Multi-Robot_Reinforcement_Learning\trained_weights\heuraid1'
leg = {1.0:'1',.85:'_85',.55:'_55',.7:'_7',.4:'_4'} #,.2:'_2',0.0:'0'
# This part is really inefficient as we load all the data every single time, but only plot one bit

## LOAD STATS
acts_rg = []
acts_rc = []
nhitss_lg = []
nhitss_lc = []
nhitss_rg = []
nhitss_rc = []
scoress_l = []
scoress_r = []
accumPtss = []
avgPtss = []
accumWinss = []
pctWinss = []

for fl,srng in leg:
    path = os.path.join(indir,'eps'+srng)
    # Load trained weights
    act_rg, act_rc, nhits_lg, nhits_lc, nhits_rg, nhits_rc, scores_l, scores_r, accumPts, avgPts, accumWins = load_metrics(path, DQNint)
    
    acts_rg.append(act_rg)
    acts_rc.append(act_rc)
    nhitss_lg.append(nhits_lg)
    nhitss_lc.append(nhits_lc)
    nhitss_rg.append(nhits_rg)
    nhitss_rc.append(nhits_rc)
    scoress_l.append(scores_l)
    scoress_r.append(scores_r)
    accumPtss.append(accumPts)
    avgPtss.append(avgPts)
    accumWinss.append(accumWins)
    pctWinss.append(pctWins)
    
    # Load tested weights
    act_rg, act_rc, nhits_lg, nhits_lc, nhits_rg, nhits_rc, scores_l, scores_r, accumPts, avgPts, accumWins = load_metrics(path, DQNint)
    
    acts_rg.append(act_rg)
    acts_rc.append(act_rc)
    nhitss_lg.append(nhits_lg)
    nhitss_lc.append(nhits_lc)
    nhitss_rg.append(nhits_rg)
    nhitss_rc.append(nhits_rc)
    scoress_l.append(scores_l)
    scoress_r.append(scores_r)
    accumPtss.append(accumPts)
    avgPtss.append(avgPts)
    accumWinss.append(accumWins)
    pctWinss.append(pctWins)


## PLOT TRAINING STATS
#1. Plot accumulated points for each epsilon value on the same plot
for each in accumPtss:
    plt.plot(each,label='eps=%.2f'%fl)

plt.xlabel('Episodes')
plt.ylabel('Points')
plt.title('Cumulative Points Scored for %s with %d Balls'%(title,noBalls))
pts.legend()
plt.show()
plt.savefig(os.path.join(savedir,'%s_accum_pts.png'%title))

#2. 

## PLOT TRAINED MODEL TEST STATS
#1. Histogram of total percentage of wins
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json


def loadjson(path,fname): #NOTE: ONLY takes lists as input, no ndarrays
    with open(os.path.join(path,fname),'r') as input: 
        data = json.load(input)
    return data
    
def everything(args):
    # Process Inputs
    # NOTE: uncomment if you aren't Alec
    # indir = './trained_weights/'+args.indir
    # savedir = './trained_weights/'+args.savedir
    indir = args.indir
    title = '70_30-Pretrained-Selfplay' # A nice, short title
    indir = 'C:/Users/Alec/Documents/GitHub/Multi-Robot_Reinforcement_Learning/trained_weights/'+indir

    # side = args.side
    noPaddles = args.numPaddles
    noBalls = args.numBalls
    DQNint = args.DQNint

    # Process data, where n = no. episodes
    scores_both = []
    for side in ['right','left']:
        no_actions_both.append(np.array(loadjson(indir,'selfplay-no-actions-'+side+'.json'))) # is an nx2 with actions for only two paddles [Rgoal, Rcenter]
    # no_hits = np.array(loadjson(indir,'no-hits.json')) # is an nx4, with no. hits for all paddles... order is [Lgoal, Lcenter, Rgoal, Rcenter]
    scores = np.array(loadjson(indir,'selfplay-scores.json')) #first is right side, second is left

    # print("ACTIONS:",loadjson(indir,'no-actions.json'))

    # act_rg = no_actions[:,0]
    # act_rc = no_actions[:,1]
    act_Lc = scores_both[1][:,0] # This is RL agent
    act_Lr = scores_both[1][:,1]
    scores_Rl = scores_both[0][:,0]
    scores_Rr = scores_both[0][:,1] # This is RL agent

    # nhits_lg = no_hits[:,0]
    # nhits_lc = no_hits[:,1]
    # nhits_rg = no_hits[:,2]
    # nhits_rc = no_hits[:,3]

    scores_L = scores[:,0] # both of these are RL agents
    scores_R = scores[:,1]

    # Figure out how many points are being scored over time, what the average is every ten games, and number of wins
    totPts_R = 0
    accumPts_R = []
    avgPts_R = []
    totWins_R=0
    accumWins_R = []
    i = 1 # Episode counter
    for eachgame in scores_R:
        totPts_R += eachgame
        if eachgame == 10:
            totWins_R += 1
        accumPts_R.append(totPts_R)
        accumWins_R.append(totWins_R)
        
        if i%DQNint==0: # If episode is a multiple of DQNint, take the average of the previous ten episodes
            avg = (np.mean(scores_R[i-DQNint:i]))
            avgPts_R.append(avg)
        i+=1
    
    accumPts_R = np.array(accumPts_R)
    accumWins_R = np.array(accumWins_R)
    avgPts_R = np.array(avgPts_R)
    
    pctWins_R = totWins_R/len(scores)
    
    totPts_L = 0
    accumPts_L = []
    avgPts_L = []
    totWins_L=0
    accumWins_L = []
    i = 1 # Episode counter
    for eachgame in scores_L:
        totPts_L += eachgame
        if eachgame == 10:
            totWins_L += 1
        accumPts_L.append(totPts_L)
        accumWins_L.append(totWins_L)
        
        if i%DQNint==0: # If episode is a multiple of DQNint, take the average of the previous ten episodes
            avg = (np.mean(scores_L[i-DQNint:i]))
            avgPts_L.append(avg)
        i+=1
    
    accumPts_L = np.array(accumPts_L)
    accumWins_L = np.array(accumWins_L)
    avgPts_L = np.array(avgPts_L)
    
    pctWins_L = totWins_L/len(scores)
    
    # Plot no. actions per episode ... pretty unintelligable at 10000 episodes
    # plt.plot(act_rg,label='Goalie')
    # plt.plot(act_rc,label='Center')
    # plt.xlabel('Episodes')
    # plt.ylabel('No. Actions')
    # plt.title('Actions per Episode for %s with %d Balls'%(title,noBalls))
    # plt.legend()
    # plt.savefig(os.path.join(indir,'%s_actions.png'%title))
    # plt.show()

    # # Plot number of hits per episode
    # plt.plot(nhits_rg,label='RGoalie')
    # plt.plot(nhits_rc,label='RCenter')
    # plt.plot(nhits_lg,label='LGoalie')
    # plt.plot(nhits_lc,label='LCenter')
    # plt.xlabel('Episodes')
    # plt.ylabel('No. Ball Hits')
    # plt.title('Cumulative Ball Hits for %s with %d Balls'%(title,noBalls))
    # plt.legend()
    # plt.savefig(os.path.join(indir,'%s_hits.png'%title))
    # plt.show()

    # Plot total number of points as episodes progress
    plt.plot(accumPts_R,label='right')
    plt.plot(accumPts_L,label='left')
    plt.xlabel('Episodes')
    plt.ylabel('Points')
    plt.title('Cumulative Points Scored for %s with %d Balls'%(title,noBalls))
    plt.legend()
    plt.savefig(os.path.join(indir,'%s_accum_pts.png'%title))
    plt.show()

    # Plot average points scored per training
    plt.plot(avgPts_R,label='right')
    plt.plot(avgPts_L,label='left')
    plt.xlabel('No. Times Trained (#episodes/%d)'%DQNint)
    plt.ylabel('Points')
    plt.title('Average Points Scored over training for %s with %d Balls'%(title,noBalls))
    plt.legend()
    plt.savefig(os.path.join(indir,'%s_avg_pts.png'%title))
    plt.show()

    # plt.plot(scores_r)
    # plt.xlabel('Episodes')
    # plt.ylabel('Points')
    # plt.suptitle('Points Scored per Episode for %s with %d Paddles, %d Balls'%(title,noPaddles,noBalls))
    # plt.title('Pct. won: %.1f'%pctWins) # This part only really makes sense if you are testing trained models
    # plt.show()
    # plt.savefig(os.path.join(indir,'%s_score.png'%title))


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--numPaddles', default = 2,type=int) # Total number of paddles in a team (e.g. if --numPaddles == 2, we have two on each team)
    parser.add_argument('--numBalls', default = 4,type=int) # Total number of balls in the game
    parser.add_argument('--indir',default = 'shashvat_selfplay',type=str) # If want to load weights from a specific subdirectory... defaults to the latest training (saved in trained_weights/latest)
    parser.add_argument('--savedir',default = 'shashvat_selfplay',type=str) # Specify directory to save selected stats in (will save everything, including weights, to trained_weights/<name>)
    parser.add_argument('--DQNint', default = 10,type=int) # How many episodes to wait between training DQN
    # parser.add_argument('--side',default = 'left',type=str) # what side pretrained network
    

    args = parser.parse_args()
    
    
    everything(args) #nhits_lg, nhits_lc, nhits_rg, nhits_rc, 
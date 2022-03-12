import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json


def loadjson(path,fname): #NOTE: ONLY takes lists as input, no ndarrays
    with open(os.path.join(path,fname),'r') as input: 
        data = json.load(input)
    return data
    
def load_metrics(indir,DQNint):
    # Process data, where n = no. episodes
    no_actions = np.array(loadjson(indir,'no-actions.json')) # is an nx2 with actions for only two paddles [Rgoal, Rcenter]
    no_hits = np.array(loadjson(indir,'no-hits.json')) # is an nx4, with no. hits for all paddles... order is [Lgoal, Lcenter, Rgoal, Rcenter]
    scores = np.array(loadjson(indir,'scores.json')) #[L,R]


    # print("ACTIONS:",loadjson(indir,'no-actions.json'))

    act_rg = no_actions[:,0]
    act_rc = no_actions[:,1]

    nhits_lg = no_hits[:,0]
    nhits_lc = no_hits[:,1]
    nhits_rg = no_hits[:,2]
    nhits_rc = no_hits[:,3]

    scores_l = scores[:,0]
    scores_r = scores[:,1]

    # Figure out how many points are being scored over time, what the average is every ten games, and number of wins
    totPts = 0
    accumPts = []
    avgPts = []
    totWins=0
    accumWins = []
    i == 1 # Episode counter
    for eachgame in scores_r:
        totPts += eachgame
        if eachgame == 10:
            totWins += 1
        accumPts.append(totPts)
        accumWins.append(totWins)
        
        if i%DQNint==0: # If episode is a multiple of DQNint, take the average of the previous ten episodes
            avg = (np.mean(scores_r[i-DQNint:i]))
            avgPts.append(avg)
        i+=1
    
    accumPts = np.array(accumPts)
    accumWins = np.array(accumWins)
    avgPts = np.array(avgPts)
    
    pctWins = wins/len(scores)
    
    return act_rg, act_rc, nhits_lg, nhits_lc, nhits_rg, nhits_rc, scores_l, scores_r, accumPts, avgPts, pctWins # Outputs are all np arrays

def plot_metrics(indir,act_rg, act_rc, nhits_lg, nhits_lc, nhits_rg, nhits_rc, scores_l, scores_r, accumPts, avgPts, accumWins,noPaddles,noBalls,DQNint):
    # Plot no. actions per episode
    plt.plot(act_rg,label='Goalie')
    plt.plot(act_rc,label='Center')
    plt.xlabel('Episodes')
    plt.ylabel('No. Actions')
    plt.title('Actions per Episode for %s with %d Balls'%(title,noBalls))
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(indir,'%s_actions.png'%title))

    # Plot number of hits per episode
    plt.plot(nhits_rg,label='RGoalie')
    plt.plot(nhits_rc,label='RCenter')
    plt.plot(nhits_lg,label='LGoalie')
    plt.plot(nhits_lc,label='LCenter')
    plt.xlabel('Episodes')
    plt.ylabel('No. Ball Hits')
    plt.title('Cumulative Ball Hits for %s with %d Balls'%(title,noBalls))
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(indir,'%s_hits.png'%title))

    # Plot total number of points as episodes progress
    plt.plot(accumPts)
    plt.xlabel('Episodes')
    plt.ylabel('Points')
    plt.title('Cumulative Points Scored for %s with %d Balls'%(title,noBalls))
    plt.show()
    plt.savefig(os.path.join(indir,'%s_accum_pts.png'%title))

    # Plot average points scored per training
    plt.plot(avgPts)
    plt.xlabel('No. Times Trained (#episodes/%d)'%DQNint)
    plt.ylabel('Points')
    plt.title('Average Points Scored over training for %s with %d Balls'%(title,noBalls))
    plt.show()
    plt.savefig(os.path.join(indir,'%s_avg_pts.png'%title))

    # plt.plot(scores_r)
    # plt.xlabel('Episodes')
    # plt.ylabel('Points')
    # plt.suptitle('Points Scored per Episode for %s with %d Paddles, %d Balls'%(title,noPaddles,noBalls))
    # plt.title('Pct. won: %.1f'%pctWins) # This part only really makes sense if you are testing trained models
    # plt.show()
    # plt.savefig(os.path.join(indir,'%s_score.png'%title))


if __name__='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--numPaddles', default = 2,type=int) # Total number of paddles in a team (e.g. if --numPaddles == 2, we have two on each team)
    parser.add_argument('--numBalls', default = 4,type=int) # Total number of balls in the game
    parser.add_argument('--indir',default = 'latest/',type=str) # If want to load weights from a specific subdirectory... defaults to the latest training (saved in trained_weights/latest)
    parser.add_argument('--savedir',default = 'latest/',type=str) # Specify directory to save selected stats in (will save everything, including weights, to trained_weights/<name>)
    parser.add_argument('--DQNint', default = 10,type=int) # How many episodes to wait between training DQN

    args = parser.parse_args()
    
    # Process Inputs
    # NOTE: uncomment if you aren't Alec
    # indir = './trained_weights/'+args.indir
    # savedir = './trained_weights/'+args.savedir
    indir = args.indir
    title = indir # A nice, short title
    indir = 'C:/Users/Alec/Documents/GitHub/Multi-Robot_Reinforcement_Learning/trained_weights/'+indir

    noPaddles = args.numPaddles
    noBalls = args.numBalls
    DQNint = args.DQNint
    
    act_rg, act_rc, nhits_lg, nhits_lc, nhits_rg, nhits_rc, scores_l, scores_r, accumPts, avgPts, accumWins = load_metrics(indir, DQNint)
    plot_metrics(indir, act_rg, act_rc, nhits_lg, nhits_lc, nhits_rg, nhits_rc, scores_l, scores_r, accumPts, avgPts, accumWins, noPaddles, noBalls, DQNint)
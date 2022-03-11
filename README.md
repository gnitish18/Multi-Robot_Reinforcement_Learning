# Multi-Robot_Reinforcement_Learning
AA277 - Multi-Robot Control and Distributed Optimization

##NOTE##
The below is a tad out of date, reference Shashvat's new branch to check it out!
Long story short, you can't pass booleans via command line as they are read as strings ... for some reason, this doesn't result in an error from argparse even when type=bool!!!


To start training with the "mod2" files:
python pong_episode_run_mod2.py

What it does:
1. starts running 1000 episodes(games)
2. collects memories up to 50000 samples and deletes the oldest to maintain that size
3. after memories is at least 50000, it pauses to retrain the model every 25 episodes
4. Takes a random collection of 10000 samples from the memories to train on
5. The right-team's movement choices gradually move from using the deterministic "closest-ball" method to using the model (by episode 200 it's all on the model)
6. Saves most recent DQN weights in ./trained_weights/latest (this will be overwitten unless new subdirectory specified in arguments, see below)
7. Saves score and number of actions per episode in ./trained_weights/latest

To test most recent training run:
python pong_episode_test_mod2.py

What it does:
1. Loads pretrained weights (by default, these come from the /latest dir, but can specify in arguments
2. Runs for 100 episodes
3. Saves full memories of <S,A,R,S'> for each episode


Can customize further with the following command line args (below are the defaults for the run script, NOT test)
1. parser.add_argument('--eps', default = 1.0,type=float) # Epsilon, initial percentage of exploratory behavior
2. parser.add_argument('--epsDecay', default = 0.005,type=float) # Epsilon decay
3. parser.add_argument('--yesRender', default = False,type=bool)
4. parser.add_argument('--withTFmodel', default = True,type=bool)
5. parser.add_argument('--noEps', default = 1000,type=int) # Number of Episodes
6. parser.add_argument('--stw', default = 10,type=int) # Score to win
7. parser.add_argument('--memBufLen', default = 50000,type=int) # Max length of memory buffer
8. parser.add_argument('--gamma', default = .95,type=float) # Discount for TD loss
9. parser.add_argument('--lr', default = .000005,type=float) # Learning rate of DQN
10. parser.add_argument('--lrDecay', default = .25,type=float) # Decay rate of DQN lr, per training*implement as per epoch?
11. parser.add_argument('--DQNint', default = 10,type=int) # How many episodes to wait between training DQN
12. parser.add_argument('--epch',default=15,type=int)
13. parser.add_argument('--batSize',default = 10,type=int)
14. parser.add_argument('--trainSetSize',default = 10000,type=int)
15. parser.add_argument('--TRAIN',default = True,type=bool) # True if training... changes some saving/loading options
16. parser.add_argument('--pretrain',default = False,type=bool) # If using pretrained weights
17. parser.add_argument('--indir',default = 'latest/',type=str) # If want to load weights from a specific subdirectory... defaults to the latest training (saved in trained_weights/latest)
18. parser.add_argument('--savedir',default = 'latest/',type=str) # Specify directory to save selected stats in (will save everything, including weights, to trained_weights/<name>)
    

    

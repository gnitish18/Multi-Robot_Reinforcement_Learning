# Multi-Robot_Reinforcement_Learning
AA277 - Multi-Robot Control and Distributed Optimization


To start training with the coupled files: 
"python pong_episode_run_coupled.py --eps 1.0 --yesRender false --withTFmodel true"

What it does:
1. starts running 1000 episodes(games)
2. collects memories up to 50000 samples and deletes the oldest to maintain that size
3. after memories is at least 50000, it pauses to retrain the model every 25 episodes
4. Takes a random collection of 10000 samples from the memories to train on
5. The right-team's movement choices gradually move from using the deterministic "closest-ball" method to using the model (by episode 200 it's all on the model)

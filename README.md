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

I'm using the _coupled files to train and the _test file to run the sim from the most recently saved weights

Things I'm trying:
1. Small batch sizes seem to work better ~10 has given me best results so far
2. learning rate decays with each training seesion
3. I'm doing smaller epochs (~15) but training more often in the loop (~every 10 episodes)
4. Added small reward for balls moving in our right half plane (+1 for moving away from our goal line and -1 for moving towards)
5. Smaller layers in the model seem to work better than larger layers (unless maybe we train longer...)

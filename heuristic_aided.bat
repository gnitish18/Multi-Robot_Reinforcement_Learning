:: Run from C:\Users\Alec\AppData\Local\Programs\Python\Python37\Scripts

FOR %%i in (1,2,3) DO (
python C:\Users\Alec\Documents\GitHub\Multi-Robot_Reinforcement_Learning\pong_episode_run_mod3.py --train True --huber True --savedir heuraid%%i/eps1 --eps 1.0 --epsDecay 0

python C:\Users\Alec\Documents\GitHub\Multi-Robot_Reinforcement_Learning\pong_episode_run_mod3.py --train True --huber True --savedir heuraid%%i/eps_85 --eps .85 --epsDecay 0

python C:\Users\Alec\Documents\GitHub\Multi-Robot_Reinforcement_Learning\pong_episode_run_mod3.py --train True --huber True --savedir heuraid%%i/eps_7 --eps .7 --epsDecay 0

python C:\Users\Alec\Documents\GitHub\Multi-Robot_Reinforcement_Learning\pong_episode_run_mod3.py --train True --huber True --savedir heuraid%%i/eps_55 --eps .55 --epsDecay 0

python C:\Users\Alec\Documents\GitHub\Multi-Robot_Reinforcement_Learning\pong_episode_run_mod3.py --train True --huber True --savedir heuraid%%i/eps_4 --eps .4 --epsDecay 0

python C:\Users\Alec\Documents\GitHub\Multi-Robot_Reinforcement_Learning\pong_episode_run_mod3.py --train True --huber True --savedir heuraid%%i/eps_2 --eps .2 --epsDecay 0

python C:\Users\Alec\Documents\GitHub\Multi-Robot_Reinforcement_Learning\pong_episode_run_mod3.py --train True --huber True --savedir heuraid%%i/eps0 --eps 0 --epsDecay 0
)
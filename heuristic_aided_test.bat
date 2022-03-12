:: Run from C:\Users\Alec\AppData\Local\Programs\Python\Python37\Scripts
:: ,_2,0
:: This will play 50 episodes with each model and save the data
FOR %%e in (_4,1,_85,_7,_55) DO (
python C:\Users\Alec\Documents\GitHub\Multi-Robot_Reinforcement_Learning\pong_episode_test_mod3.py --train False --indir heuraid1/eps%%e --savedir heuraid1/eps%%e_test --yesRender False --noEps 50
)
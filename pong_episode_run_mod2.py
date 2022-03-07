
import pygame, sys, time, random, os
from pygame.locals import *
import argparse
import math
import numpy as np
import pickle

import tensorflow as tf
from objectClasses_mod2 import *
from train_from_memories_mod2 import *

white = [255, 255, 255]
black = [0, 0, 0]
clock = pygame.time.Clock()

# NOTE: numbers need to be in native python 'float', NOT numpy.float32 (use .item() to convert)
def write2json(data,path,fname): #NOTE: ONLY takes lists as input, no ndarrays
    if not os.path.exists(path): # Create dir if doesn't already exist
        os.makedirs(path)
    with open(os.path.join(path,fname),'w') as output: # writing 'wb' here screws this mess up royally:
                                                        # Raises "a bytes-like object is required, not 'str'", even when using '.tolist()' on component matrix
        json.dump(data,output)
        # pickle.dump(data,output)
        # output.write(json.dumps(data))

class foosPong_model(tf.keras.Model):
    def __init__(self):
        super(foosPong_model, self).__init__()
        ###############################################
        self.drop = tf.keras.layers.Dropout(0.2)
        self.gauss = tf.keras.layers.GaussianNoise(stddev=0.1)
        self.n1 = tf.keras.layers.BatchNormalization()
        #self.n2 = tf.keras.layers.BatchNormalization()
        
        self.d1 = tf.keras.layers.Dense(28, activation='relu')
        self.d2 = tf.keras.layers.Dense(28*4, activation='relu')
#        self.d3 = tf.keras.layers.Dense(48*8, activation='relu')
#        self.d4 = tf.keras.layers.Dense(48*8, activation='relu')
        # self.d5 = tf.keras.layers.Dense(28*4, activation='relu')
        self.d6 = tf.keras.layers.Dense(28, activation='relu')
        self.d7 = tf.keras.layers.Dense(4)
        
        ###############################################
        
    def call(self, x):
        
        x = self.n1(x)
        x = self.gauss(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.drop(x)
#        x = self.d3(x)
#        x = self.drop(x)
#        x = self.d4(x)
#        x = self.drop(x)
        # x = self.d5(x)
        # x = self.drop(x)
        x = self.d6(x)
        return self.d7(x)

def game_loop(screen, paddles, balls, table_size, clock_rate, turn_wait_rate, score_to_win, display, eps, yesRender, withTFmodel, TRAIN): # Make sure that epsilon is processed properly here, reset defaults
    # Count both paddles (by team) and balls
    nos = [] # List containing total numbers
    noPaddles = [0,0] # List to store number of paddles one each team
    for eachPad in paddles: # Check which direction paddles are facing and add to team count
        if eachPad.facing == 0:
            noPaddles[1] += 1 # RHS team
        else:
            noPaddles[0] += 1 # LHS team
            
    nos.append(noPaddles)
    nos.append([len(balls)])
    
    states = [] #state of all paddles and all balls, positions and velocities
    actions = [] #actions that each paddle takes
    rewards = [] #sum of rewards after each movement
    next_states = []

    idx = 0
    score = [0, 0]
    while max(score) < score_to_win:
        old_score = score[:]
        #print(idx)
        
        #balls, score = check_point(score, balls, table_size)
        
        ########### update memories with current states of paddles and balls ############################################################
        
        curr_states = []
        for paddle in paddles:
            curr_states.append(paddle.frect.pos[0])
            curr_states.append(paddle.frect.pos[1])
        for ball in balls:
            curr_states.append(ball.get_center()[0])
            curr_states.append(ball.get_center()[1])
            curr_states.append(ball.speed[0])
            curr_states.append(ball.speed[1])
        
       
        # Take actions...and add to memory actions
        curr_actions = []
        for i in range(len(paddles)):
            if paddles[i].facing == 0:
                action = paddles[i].move(i, paddles, balls, table_size, curr_states, withTFmodel, eps, TRAIN)
                curr_actions.append(action)
            else:
                action = paddles[i].move(i, paddles, balls, table_size, curr_states, False, eps, TRAIN)
        
        
        
        for ball in balls:
            paddled = 0
            inv_move_factor = int((ball.speed[0]**2+ball.speed[1]**2)**.5)
            if inv_move_factor > 0:
                for i in range(inv_move_factor):
                    paddled = ball.move(paddles, table_size, 1./inv_move_factor)
            else:
                paddled = ball.move(paddles, table_size, 1)
                
            if paddled == 1:
                ball.lastPaddleIdx = idx
            
       
        new_states = []
        for paddle in paddles:
            new_states.append(paddle.frect.pos[0])
            new_states.append(paddle.frect.pos[1])
        for ball in balls:
            new_states.append(ball.get_center()[0])
            new_states.append(ball.get_center()[1])
            new_states.append(ball.speed[0])
            new_states.append(ball.speed[1])
        
        
        # Check if a ball scored and add rewards accordingly, so rewards[i] should correspond to actions taken at actions[i]
        balls, score, lastPaddleIdxs = check_point(score, balls, table_size)
        
        curr_rewards = []
        if score != old_score:
            if score[0] != old_score[0]:
                #-1 for each point opponent scores
                curr_rewards.append(-10)
                curr_rewards.append(-10)
#                for i in lastPaddleIdxs:
#                    if i != -1:
#                        rewards[i][0] = rewards[i][0] - 100
#                        rewards[i][1] = rewards[i][1] - 100
            else:
                #+1 each time our team scores
                curr_rewards.append(10)
                curr_rewards.append(10)
                for i in lastPaddleIdxs:
                    if i != -1:
                        rewards[i][0] = rewards[i][0] + 20
                        rewards[i][1] = rewards[i][1] + 20
        else:
            #check ball locations and movements
            ball_loc_reward = 0
            for ball in balls:
                if ball.get_center()[0] > table_size[0]/2:
                    if ball.speed[0] > 0: ball_loc_reward - 1
                    else: ball_loc_reward + 1
            # Reward 0 if nothing happens?
            curr_rewards.append(ball_loc_reward)
            curr_rewards.append(ball_loc_reward) 
        
        
        states.append(curr_states)
        actions.append(curr_actions)
        next_states.append(new_states)
        rewards.append(curr_rewards)
        idx = idx + 1

################       SCREEN RENDER       ########################

        if yesRender:
        # if False:
            render(screen, paddles, balls, score, table_size)

##########################################################################

    for i in range(len(balls)):
        balls[i] = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
    
    print(score)
    
    #print("idx", idx)
    print("states: ", len(states), "actions: ", len(actions), "rewards: ", len(rewards), "next_states: ", len(next_states))
    return states, actions, rewards, next_states, score # Episodal results


def init_game(args):
    # Define arguments
    eps = float(args.eps)
    episodes = args.noEps
    eps_decay = args.epsDecay
    gamma = args.gamma
    lr = args.lr
    lr_decay = args.lrDecay
    mbuf_len = args.memBufLen
    DQNint = args.DQNint
    epochs = args.epch
    batch_size = args.batSize
    train_set_size = args.trainSetSize
    TRAIN = args.TRAIN
    yesRender = args.yesRender
    withTFmodel = args.withTFmodel
    pretrain = args.pretrain
    indir = './trained_weights/'+args.indir
    savedir = './trained_weights/'+args.savedir
    

    table_size = (800, 400)
    paddle_size = (5, 70)
    ball_size = (15, 15)
    paddle_speed = 5 #1
    max_angle = 45

    paddle_bounce = 1.5 #1.2
    wall_bounce = 1.00
    dust_error = 0.00
    init_speed_mag = 2
    timeout = 0.0003
    clock_rate = 200 #80
    turn_wait_rate = 3
    score_to_win = args.stw

    screen = pygame.display.set_mode(table_size)
    pygame.display.set_caption('PongAIvAI')

    paddles = [Paddle((30, table_size[1]/4), paddle_size, .5*paddle_speed, max_angle,  1, timeout, 0), \
               Paddle((table_size[0] - 30, table_size[1]/4), paddle_size, .5*paddle_speed, max_angle,  0, timeout, 0), \
               Paddle((table_size[0] - 300, table_size[1] - table_size[1]/4), paddle_size, .5*paddle_speed, max_angle, 0, timeout, 1)]
               
    #ball = Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)
    balls = [Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag), Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)]
    
    # Count both paddles (by team) and balls
    nos = [] # List containing total numbers
    noPaddles = [0,0] # List to store number of paddles one each team
    for eachPad in paddles: # Check which direction paddles are facing and add to team count
        if eachPad.facing == 0:
            noPaddles[1] += 1 # RHS team
        else:
            noPaddles[0] += 1 # LHS team
            
    nos.append(noPaddles)
    noBalls = len(balls) # A scalar, number of balls
    nos.append([noBalls]) # Put in brackets since noPaddles is a list
    totalPaddles = sum(noPaddles) # Total number of paddles on the table
    # Add trainable number of paddles here*, dimstate here
    
    def pong_ai(paddle_frect, ball_frect, table_size):
        if paddle_frect.pos[1] + paddle_frect.size[1]/2 < ball_frect.pos[1] + ball_frect.size[1]/2:
           return "down"
        else:
           return  "up"
    
    
    def foosPong_ai(states,eps,yesRender,withTFmodel, totalPaddles, noBalls, id): # Do we need eps here?*
        dimstate = 2*totalPaddles + 4*noBalls # Computed dimension of state space based on how many things
        output = foosPong(np.asarray(states, dtype='float32').reshape((1,dimstate)))
        noActions = 2 # CHANGE WHEN DOING NO-OP
        team_Q_values = tf.reshape(output, [noActions,noActions])
        action_idx = tf.math.argmax(team_Q_values[id,:]).numpy()
        
        if action_idx == 0:
            return "down"
        else:
            return "up"
    
    def move_getter(withTFmodel, eps, states, id, paddle_frect, ball_frect, table_size, nos, TRAIN):
        if withTFmodel:
            if TRAIN: # Only use epsilon-greedy for training
                if np.random.random() < eps:
                    return pong_ai(paddle_frect, ball_frect, table_size)
                else:
                    return foosPong_ai(states,eps,yesRender,withTFmodel, totalPaddles, noBalls, id)
            else: # If not training (testing), return pure model output
                return foosPong_ai(states,eps,yesRender,withTFmodel, totalPaddles, noBalls, id)
        else:
            return pong_ai(paddle_frect, ball_frect, table_size)
    
    
    # Set move getter methods/functions for all paddles
    for eachPad in paddles:
        eachPad.move_getter = move_getter
    
    # Migrate into game and DQN training setup    
    # If not training, load recently trained weights
    if withTFmodel:
        foosPong = foosPong_model()
        if pretrain: # If loading pretrained weights
            foosPong.load_weights(indir+'foosPong_model_integrated')
    
    memory_states = []
    memory_actions = []
    memory_rewards = []
    memory_next_states = []
    no_actions = []
    scores = []
    
    for ep in range(episodes):
        print(f"\nEpisode: {ep}")
        ep_states, ep_actions, ep_rewards, ep_next_states, ep_score = game_loop(screen, paddles, balls, table_size, clock_rate, turn_wait_rate, score_to_win, 1, eps-eps_decay*ep, yesRender, withTFmodel, TRAIN)
        
        # Compute total of actions for each paddle * MAY NEED TO CHANGE SIZE FOR DIFF NUM
        pad_act = np.array(ep_actions) # Convert to numpy for easy summing along rows
        ep_no_actions = np.sum(pad_act,axis=0).tolist() # To convert back from np.float32 to 'float' for json saving
        
        memory_states = memory_states + ep_states
        memory_actions = memory_actions + ep_actions
        memory_rewards = memory_rewards + ep_rewards
        memory_next_states = memory_next_states + ep_next_states
        scores.append(ep_score)
        
        no_actions.append(ep_no_actions)
        print("memory_states: ", len(memory_states), "memory_actions: ", len(memory_actions), "memory_rewards: ", len(memory_rewards), "memory_next_states: ", len(memory_next_states), "\n")
        
        # after so many steps, take a pause
        # foosPong_model = train_nn(memories, foosPong_model)
        if TRAIN: # If train flag, then periodically truncate memory buffer and save training data
            if len(memory_states) > mbuf_len:
                if ep % DQNint == 0: # Some weird behavior here, star
                    memories = [np.asarray(memory_states, dtype='float32'), np.asarray(memory_actions, dtype='float32'), np.asarray(memory_rewards, dtype='float32'), np.asarray(memory_next_states, dtype='float32')]
                    
                    # print(epochs)
                    # print(batch_size)
                    # print(train_set_size)
                    foosPong = train_nn(lr, memories, foosPong, foosPong, gamma, epochs, batch_size, train_set_size, totalPaddles, noBalls, savedir)
                    lr = lr*(1 - lr_decay) # Decay learning rate
                
                del memory_states[0:len(ep_states)]
                del memory_actions[0:len(ep_actions)]
                del memory_rewards[0:len(ep_rewards)]
                del memory_next_states[0:len(ep_next_states)]
                
            # NOTE: Training function already saves its metrics and weights
            write2json(scores,savedir,fname="scores.json")
            write2json(no_actions,savedir,fname="no-actions.json")  # saving total number of actions for each of the trained paddles for each episode
            README = "Episode: %d, can save other identifying features here" %(ep+1)
            write2json(README,savedir,fname="README.txt")
            # write2json(no_bounces,savedir,fname="no-bounces.json")
        else: # If not training, then testing: don't delete anything and save at the end!
              # NOTE: This saves for every episode
            write2json(memory_states,savedir,fname="memory_states.json")
            print("States dumped...")
            
            write2json(memory_actions,savedir,fname="memory_actions.json")
            print("Actions dumped...")
            
            write2json(memory_rewards,savedir,fname="memory_rewards.json")
            print("Rewards dumped...")
            
            write2json(memory_next_states,savedir,fname="memory_next_states.json")
            print("Next_states dumped...")
            
            print(np.asarray(memory_states).shape)
            
            print("All saved to trained_weights/"+savedir+"...")
        
        
    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eps', default = 1.0,type=float) # Epsilon, initial percentage of exploratory behavior
    parser.add_argument('--epsDecay', default = 0.005,type=float) # Epsilon decay
    parser.add_argument('--yesRender', default = False,type=bool)
    parser.add_argument('--withTFmodel', default = True,type=bool)
    parser.add_argument('--noEps', default = 1000,type=int) # Number of Episodes
    parser.add_argument('--stw', default = 10,type=int) # Score to win
    parser.add_argument('--memBufLen', default = 50000,type=int) # Max length of memory buffer
    parser.add_argument('--gamma', default = .95,type=float) # Discount for TD loss
    parser.add_argument('--lr', default = .000005,type=float) # Learning rate of DQN
    parser.add_argument('--lrDecay', default = .25,type=float) # Decay rate of DQN lr, per training*implement as per epoch?
    parser.add_argument('--DQNint', default = 10,type=int) # How many episodes to wait between training DQN
    parser.add_argument('--epch',default=15,type=int)
    parser.add_argument('--batSize',default = 10,type=int)
    parser.add_argument('--trainSetSize',default = 10000,type=int)
    parser.add_argument('--TRAIN',default = True,type=bool) # True if training... changes some saving/loading options
    parser.add_argument('--pretrain',default = False,type=bool) # If using pretrained weights
    parser.add_argument('--indir',default = 'latest/',type=str) # If want to load weights from a specific subdirectory... defaults to the latest training (saved in trained_weights/latest)
    parser.add_argument('--savedir',default = 'latest/',type=str) # Specify directory to save selected stats in (will save everything, including weights, to trained_weights/<name>)
    
    args = parser.parse_args()
    
    pygame.init()
    init_game(args)

# Save to separate folders when training and testng
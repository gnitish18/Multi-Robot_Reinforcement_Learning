
import pygame, sys, time, random, os
from pygame.locals import *
import argparse
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from objectClasses_coupled import *
from train_from_memories_coupled import *
import json
white = [255, 255, 255]
black = [0, 0, 0]
clock = pygame.time.Clock()

def write2json(data,path,fname): #NOTE: ONLY takes lists as input, no ndarrays
    if not os.path.exists(path): # Create dir if doesn't already exist
        os.makedirs(path)
    with open(os.path.join(path,fname),'w') as output: # writing 'wb' here screws this mess up royally:
                                                        # Raises "a bytes-like object is required, not 'str'", even when using '.tolist()' on component matrix
        json.dump(data,output)
        # pickle.dump(data,output)
        # output.write(json.dumps(data))

class foosPong_left(tf.keras.Model):
    def __init__(self):
        super(foosPong_left, self).__init__()
        ###############################################
        self.drop = tf.keras.layers.Dropout(0.2)
        self.gauss = tf.keras.layers.GaussianNoise(stddev=0.1)
        self.n1 = tf.keras.layers.BatchNormalization()
        #self.n2 = tf.keras.layers.BatchNormalization()
        
        self.d1 = tf.keras.layers.Dense(2*24, activation='relu')
        self.d2 = tf.keras.layers.Dense(2*24*4, activation='relu')
#        self.d3 = tf.keras.layers.Dense(48*8, activation='relu')
        self.d4 = tf.keras.layers.Dense(2*24*6, activation='relu')
        self.d5 = tf.keras.layers.Dense(2*24*4, activation='relu')
        self.d6 = tf.keras.layers.Dense(2*24, activation='relu')
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
        x = self.d4(x)
        x = self.drop(x)
        x = self.d5(x)
        x = self.drop(x)
        x = self.d6(x)
        return self.d7(x)
        
class foosPong_right(tf.keras.Model):
    def __init__(self):
        super(foosPong_right, self).__init__()
        ###############################################
        self.drop = tf.keras.layers.Dropout(0.2)
        self.gauss = tf.keras.layers.GaussianNoise(stddev=0.1)
        self.n1 = tf.keras.layers.BatchNormalization()
        #self.n2 = tf.keras.layers.BatchNormalization()
        
        self.d1 = tf.keras.layers.Dense(2*24, activation='relu')
        self.d2 = tf.keras.layers.Dense(2*24*4, activation='relu')
#        self.d3 = tf.keras.layers.Dense(48*8, activation='relu')
        self.d4 = tf.keras.layers.Dense(2*24*6, activation='relu')
        self.d5 = tf.keras.layers.Dense(2*24*4, activation='relu')
        self.d6 = tf.keras.layers.Dense(2*24, activation='relu')
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
        x = self.d4(x)
        x = self.drop(x)
        x = self.d5(x)
        x = self.drop(x)
        x = self.d6(x)
        return self.d7(x)


def game_loop(screen, paddles, balls, table_size, clock_rate, turn_wait_rate, score_to_win, display, e, yesRender=True, withTFmodel=False):
    score = [0, 0]
    
    
    
    states = [] #state of all paddles and all balls, positions and velocities
    actions_left = [] #actions that each paddle takes
    actions_right = [] #actions that each paddle takes
    rewards = [] #sum of rewards after each movement
    next_states = []

    idx = 0
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
        curr_actions_left = []
        curr_actions_right = []
        for i in range(len(paddles)):
            if paddles[i].facing == 0:
                action = paddles[i].move(i, paddles, balls, table_size, curr_states, withTFmodel, e)
                curr_actions_right.append(action)
            else:
                action = paddles[i].move(i, paddles, balls, table_size, curr_states, withTFmodel, e)
                curr_actions_left.append(action)
        
        
        
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
        balls, score, lastPaddleIdxs_left, lastPaddleIdxs_right = check_point(score, balls, table_size)
        
        curr_rewards = []
        if score != old_score:
            if score[0] != old_score[0]:
                #-1 for each point opponent scores
                curr_rewards.append(-20)
                curr_rewards.append(-20)
                for i in lastPaddleIdxs_left:
                    if i != -1:
                        rewards[i][0] = rewards[i][0] - 20
                        rewards[i][1] = rewards[i][1] - 20
                for i in lastPaddleIdxs_right:
                    if i != -1:
                        rewards[i][0] = rewards[i][0] - 20
                        rewards[i][1] = rewards[i][1] - 20
            else:
                #+1 each time our team scores
                curr_rewards.append(20)
                curr_rewards.append(20)
                for i in lastPaddleIdxs_left:
                    if i != -1:
                        rewards[i][0] = rewards[i][0] + 20
                        rewards[i][1] = rewards[i][1] + 20
                for i in lastPaddleIdxs_right:
                    if i != -1:
                        rewards[i][0] = rewards[i][0] + 20
                        rewards[i][1] = rewards[i][1] + 20
        else:
            #check ball locations and movements
            ball_loc_reward = 0
            for ball in balls:
                if ball.get_center()[0] > table_size[0]/2:
                    if ball.speed[0] > 0: ball_loc_reward - 2
                    else: ball_loc_reward + 2
            # Reward 0 if nothing happens?
            curr_rewards.append(ball_loc_reward)
            curr_rewards.append(ball_loc_reward)
            
        
        
        states.append(curr_states)
        actions_left.append(curr_actions_left)
        actions_right.append(curr_actions_right)
        next_states.append(new_states)
        rewards.append(curr_rewards)
        idx = idx + 1
        


################       SCREEN RENDER       ########################

        if yesRender:
            render(screen, paddles, balls, score, table_size)

##########################################################################

    for i in range(len(balls)):
        balls[i] = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
    
    print(score)
    print("states: ", len(states))
    return states, actions_left, actions_right, rewards, next_states, score


def init_game(args):
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
    score_to_win = 10


    screen = pygame.display.set_mode(table_size)
    pygame.display.set_caption('PongAIvAI')


    paddles = [Paddle((30, table_size[1]/4), paddle_size, paddle_speed, max_angle,  1, timeout, 0), \
               Paddle((130, table_size[1] - table_size[1]/4), paddle_size, paddle_speed, max_angle,  1, timeout, 1), \
               Paddle((table_size[0] - 30, table_size[1]/4), paddle_size, paddle_speed, max_angle,  0, timeout, 0), \
               Paddle((table_size[0] - 130, table_size[1] - table_size[1]/4), paddle_size, paddle_speed, max_angle, 0, timeout, 1)]
               
    #ball = Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)
    balls = [Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag), Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag), Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag), Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)]
    
    
    
    def pong_ai(paddle_frect, ball_frect, table_size):
        #change if you want to add some brains to pong_ai
        useHeuristic = 0.0
        
        if np.random.random() < useHeuristic:
            if paddle_frect.pos[1] + paddle_frect.size[1]/2 < ball_frect.pos[1] + ball_frect.size[1]/2:
               return "down"
            else:
               return  "up"
        else:
            if np.random.random() < 0.5:
                return "down"
            else:
               return  "up"
            

    def foosPong_ai(states, id, facing):
        if facing == 1:
            output = foosPong_l(np.asarray(states, dtype='float32').reshape((1,24)))
        elif facing == 0:
            output = foosPong_r(np.asarray(states, dtype='float32').reshape((1,24)))
            
        team_Q_values = tf.reshape(output, [2,2])
        action_idx = tf.math.argmax(team_Q_values[id,:]).numpy()
        
        if action_idx == 0:
            return "down"
        else:
            return "up"
        
    def move_getter(withTFmodel, e, states, id, paddle_frect, ball_frect, table_size, facing):
        if withTFmodel:
            return foosPong_ai(states, id, facing)
        else:
            return pong_ai(paddle_frect, ball_frect, table_size)
    
    
    # Set move getter functions
    paddles[0].move_getter = move_getter
    paddles[1].move_getter = move_getter
    paddles[2].move_getter = move_getter
    paddles[3].move_getter = move_getter
        
        
        
    #foosPong = foosPong_model()
    
    foosPong_l = foosPong_left()
    foosPong_l.load_weights('./trained_weights/latest/foosPong_model_left_selfplay_v2')
    foosPong_r = foosPong_right()
    foosPong_r.load_weights('./trained_weights/latest/foosPong_model_right_selfplay_v2')
    
    eps = 0.0 #don't ever use heuristic...
    yesRender = True
    withTFmodel = False
    if args.yesRender == 'false': yesRender = False
    if args.withTFmodel == 'true': withTFmodel = True
        #foosPong.load_weights('./trained_weights/foosPong_model_v0')
       
      
    memory_states = []
    memory_actions_left = []
    memory_rewards_left = []
    memory_actions_right = []
    memory_rewards_right = []
    memory_next_states = []
    no_actions_right = []
    no_actions_left = []
    action_counts = []
    scores = []
    eps_decay = 0.005
    episodes = 1000
    lr = 0.0000025
    lr_decay = 0.25
    for ep in range(episodes):
        print(f"\nEpisode: {ep}")
        ep_states, ep_actions_left, ep_actions_right, ep_rewards, ep_next_states, ep_score = game_loop(screen, paddles, balls, table_size, clock_rate, turn_wait_rate, score_to_win, 1, eps-eps_decay*ep, yesRender=yesRender, withTFmodel=withTFmodel)
        pad_act_right = np.array(ep_actions_right) # Convert to numpy for easy summing along rows
        ep_no_actions_right = np.sum(pad_act_right,axis=0).tolist() 
        pad_act_left = np.array(ep_actions_left) # Convert to numpy for easy summing along rows
        ep_no_actions_left = np.sum(pad_act_left,axis=0).tolist() 
        memory_states = memory_states + ep_states
        memory_actions_left = memory_actions_left + ep_actions_left
        memory_actions_right = memory_actions_right + ep_actions_right
        memory_rewards_left = memory_rewards_left + list(-1*np.asarray(ep_rewards))
        memory_rewards_right = memory_rewards_right + ep_rewards
        memory_next_states = memory_next_states + ep_next_states
        scores.append(ep_score)
        no_actions_right.append(ep_no_actions_right)
        no_actions_left.append(ep_no_actions_left)
        print("Memory Size: ", len(memory_states))
        
        action_counts.append(len(ep_states))
        
        if ep % 10 == 0 and ep != 0:
            with open("action_counts.txt", "wb") as fp:
                pickle.dump(action_counts, fp)
            print("Action Counts dumped...  Current Avg: ", np.mean(np.asarray(action_counts)))
        
        # after so many steps, take a pause
        # foosPong_model = train_nn(memories, foosPong_model)
        if len(memory_states) > 100000:
            if ep % 10 == 0:
                memories_left = [np.asarray(memory_states, dtype='float32'), np.asarray(memory_actions_left, dtype='float32'), np.asarray(memory_rewards_left, dtype='float32'), np.asarray(memory_next_states, dtype='float32')]
                memories_right = [np.asarray(memory_states, dtype='float32'), np.asarray(memory_actions_right, dtype='float32'), np.asarray(memory_rewards_right, dtype='float32'), np.asarray(memory_next_states, dtype='float32')]
                
                foosPong_r = train_nn(lr, memories_right, foosPong_r, foosPong_r)
                foosPong_r.save_weights('./trained_weights/latest/foosPong_model_right_selfplay') #og was v4
                foosPong_l = train_nn(lr, memories_left, foosPong_l, foosPong_l)
                foosPong_l.save_weights('./trained_weights/latest/foosPong_model_left_selfplay') #og was leftTeam
                
                lr = lr - lr_decay*lr
                
            
            del memory_states[0:len(ep_states)]
            del memory_actions_left[0:len(ep_states)]
            del memory_actions_right[0:len(ep_states)]
            del memory_rewards_left[0:len(ep_states)]
            del memory_rewards_right[0:len(ep_states)]
            del memory_next_states[0:len(ep_states)]
            savedir = './trained_weights/latest'
            write2json(scores,savedir,fname="selfplay-scores.json")
            write2json(no_actions_right,savedir,fname="selfplay-no-actions-right.json") 
            write2json(no_actions_left,savedir,fname="selfplay-no-actions-left.json")  # saving total number of actions for each of the trained paddles for each episode
            README = "Episode: %d, can save other identifying features here" %(ep+1)
            write2json(README,savedir,fname="README.json")
    
    pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eps')
    parser.add_argument('--yesRender')
    parser.add_argument('--withTFmodel')

    args = parser.parse_args()
    
    pygame.init()
    init_game(args)

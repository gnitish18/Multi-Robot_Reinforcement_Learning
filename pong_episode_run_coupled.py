
import pygame, sys, time, random, os
from pygame.locals import *
import argparse
import math
import numpy as np
import pickle

import tensorflow as tf
from objectClasses import *
from train_from_memories_coupled import *

white = [255, 255, 255]
black = [0, 0, 0]
clock = pygame.time.Clock()

class foosPong_model(tf.keras.Model):
    def __init__(self):
        super(foosPong_model, self).__init__()
        ###############################################
        self.drop = tf.keras.layers.Dropout(0.20)
        self.gauss = tf.keras.layers.GaussianNoise(stddev=0.2)
        #self.n1 = tf.keras.layers.BatchNormalization()
        #self.n2 = tf.keras.layers.BatchNormalization()
        
        self.d1 = tf.keras.layers.Dense(48, activation='relu')
        self.d2 = tf.keras.layers.Dense(48*4, activation='relu')
        self.d3 = tf.keras.layers.Dense(48*8, activation='relu')
        self.d4 = tf.keras.layers.Dense(48*4, activation='relu')
        self.d5 = tf.keras.layers.Dense(48, activation='relu')
        
        # size 4, so that each teammate has action space of (up, down)
        # output here is Q value for each possible action for each teammate, which gets added together in loss function for total max q-value
        self.d6 = tf.keras.layers.Dense(4)
        
        ###############################################
        
    def call(self, x):
        x = self.gauss(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.drop(x)
        x = self.d3(x)
        x = self.drop(x)
        x = self.d4(x)
        x = self.drop(x)
        x = self.d5(x)
        return self.d6(x)
        




def game_loop(screen, paddles, balls, table_size, clock_rate, turn_wait_rate, score_to_win, display, e, yesRender=True, withTFmodel=False):
    score = [0, 0]
    
    
    
    states = [] #state of all paddles and all balls, positions and velocities
    actions = [] #actions that each paddle takes
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
        curr_actions = []
        for i in range(len(paddles)):
            if paddles[i].facing == 0:
                action = paddles[i].move(i, paddles, balls, table_size, curr_states, withTFmodel, e)
                curr_actions.append(action)
            else:
                action = paddles[i].move(i, paddles, balls, table_size, curr_states, False, e)
        
        
        
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
                curr_rewards.append(-50)
                curr_rewards.append(-50)
            else:
                #+1 each time our team scores
                curr_rewards.append(0)
                curr_rewards.append(0)
                for i in lastPaddleIdxs:
                    # adds reward back to the time step that a paddle on our team hit the ball
                    if i != -1:
                        #print(i)
                        #print(idx)
                        rewards[i][0] = rewards[i][0] + 100
                        rewards[i][1] = rewards[i][1] + 100
        else:
            # Reward 0 if nothing happens?
            curr_rewards.append(0)
            curr_rewards.append(0)
            
        
        
        if (np.random.random() < 1.0) or score != old_score:
            states.append(curr_states)
            actions.append(curr_actions)
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
    print("idx", idx)
    print("states: ", len(states), "actions: ", len(actions), "rewards: ", len(rewards), "next_states: ", len(next_states))
    return states, actions, rewards, next_states


def init_game(args):
    table_size = (800, 600)
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
               Paddle((300, table_size[1] - table_size[1]/4), paddle_size, paddle_speed, max_angle,  1, timeout, 1), \
               Paddle((table_size[0] - 30, table_size[1]/4), paddle_size, paddle_speed, max_angle,  0, timeout, 0), \
               Paddle((table_size[0] - 300, table_size[1] - table_size[1]/4), paddle_size, paddle_speed, max_angle, 0, timeout, 1)]
               
    #ball = Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)
    balls = [Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag), Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag), Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag), Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)]
    
    
    
    def pong_ai(paddle_frect, ball_frect, table_size):
        if paddle_frect.pos[1] + paddle_frect.size[1]/2 < ball_frect.pos[1] + ball_frect.size[1]/2:
           return "down"
        else:
           return  "up"
    
    def foosPong_ai(states, id):
        output = foosPong(np.asarray(states, dtype='float32').reshape((1,24)))
        team_Q_values = tf.reshape(output, [2,2])
        action_idx = tf.math.argmax(team_Q_values[id,:]).numpy()
        
        if action_idx == 0:
            return "down"
        else:
            return "up"
        
    def move_getter(withTFmodel, e, states, id, paddle_frect, ball_frect, table_size):
        if withTFmodel:
            if np.random.random() < e:
                return pong_ai(paddle_frect, ball_frect, table_size)
            else:
                return foosPong_ai(states, id)
        else:
            return pong_ai(paddle_frect, ball_frect, table_size)
    
    
    # Set move getter functions
    paddles[0].move_getter = move_getter
    paddles[1].move_getter = move_getter
    paddles[2].move_getter = move_getter
    paddles[3].move_getter = move_getter
        
        
        
    foosPong = foosPong_model()
    eps = float(args.eps)
    yesRender = True
    withTFmodel = False
    if args.yesRender == 'false': yesRender = False
    if args.withTFmodel == 'true': withTFmodel = True
        #foosPong.load_weights('./trained_weights/foosPong_model_v0')
       
       
       
    
    episodes = 1000
    memory_states = []
    memory_actions = []
    memory_rewards = []
    memory_next_states = []
    decay = 0.005
    for ep in range(episodes):
        print(f"\nEpisode: {ep}")
        ep_states, ep_actions, ep_rewards, ep_next_states = game_loop(screen, paddles, balls, table_size, clock_rate, turn_wait_rate, score_to_win, 1, eps-decay*ep, yesRender=yesRender, withTFmodel=withTFmodel)
        
        memory_states = memory_states + ep_states
        memory_actions = memory_actions + ep_actions
        memory_rewards = memory_rewards + ep_rewards
        memory_next_states = memory_next_states + ep_next_states
        print("memory_states: ", len(memory_states), "memory_actions: ", len(memory_actions), "memory_rewards: ", len(memory_rewards), "memory_next_states: ", len(memory_next_states), "\n")
        
        
        # after so many steps, take a pause
        # foosPong_model = train_nn(memories, foosPong_model)
        if len(memory_states) > 50000:
            if ep % 25 == 0:
                memories = [np.asarray(memory_states, dtype='float32'), np.asarray(memory_actions, dtype='float32'), np.asarray(memory_rewards, dtype='float32'), np.asarray(memory_next_states, dtype='float32')]
                
                foosPong = train_nn(memories, foosPong, foosPong)
            print("before", len(memory_states))
            del memory_states[0:len(ep_states)]
            del memory_actions[0:len(ep_actions)]
            del memory_rewards[0:len(ep_rewards)]
            del memory_next_states[0:len(ep_next_states)]
            print("after", len(memory_states))
        
        
        
        
#    with open("memory_states.txt", "wb") as fp:
#        pickle.dump(memory_states, fp)
#    print("States dumped...")
#
#    with open("memory_actions.txt", "wb") as fp:
#        pickle.dump(memory_actions, fp)
#    print("Actions dumped...")
#
#    with open("memory_rewards.txt", "wb") as fp:
#        pickle.dump(memory_rewards, fp)
#    print("Rewards dumped...")
#
#    with open("memory_next_states.txt", "wb") as fp:
#        pickle.dump(memory_next_states, fp)
#    print("Next_states dumped...")
#
#    print(np.asarray(memory_states).shape)
    
    pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eps')
    parser.add_argument('--yesRender')
    parser.add_argument('--withTFmodel')

    args = parser.parse_args()
    
    pygame.init()
    init_game(args)


import pygame, sys, time, random, os
from pygame.locals import *

import math
import numpy as np
import pickle

import tensorflow as tf
from objectClasses import *

white = [255, 255, 255]
black = [0, 0, 0]
clock = pygame.time.Clock()

class foosPong_model(tf.keras.Model):
    def __init__(self):
        super(foosPong_model, self).__init__()
        ###############################################
        self.drop = tf.keras.layers.Dropout(0.2)
        self.gauss = tf.keras.layers.GaussianNoise(stddev=0.1)
        self.n1 = tf.keras.layers.BatchNormalization()
        #self.n2 = tf.keras.layers.BatchNormalization()
        
        self.d1 = tf.keras.layers.Dense(48, activation='relu')
        self.d2 = tf.keras.layers.Dense(48*4, activation='relu')
#        self.d3 = tf.keras.layers.Dense(48*8, activation='relu')
#        self.d4 = tf.keras.layers.Dense(48*8, activation='relu')
        self.d5 = tf.keras.layers.Dense(48*4, activation='relu')
        self.d6 = tf.keras.layers.Dense(48, activation='relu')
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
        x = self.d5(x)
        x = self.drop(x)
        x = self.d6(x)
        return self.d7(x)
        




def game_loop(screen, paddles, balls, table_size, clock_rate, turn_wait_rate, score_to_win, display, eps=0.3, yesRender=True, withTFmodel=False):

    score = [0, 0]
    
    states = [] #state of all paddles and all balls, positions and velocities
    actions = [] #actions that each paddle takes
    rewards = [] #sum of rewards after each movement
    next_states = []

    while max(score) < score_to_win:
        old_score = score[:]
        
        
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
                action = paddles[i].move(i, paddles, balls, table_size, curr_states, withTFmodel, 1.0)
                curr_actions.append(action)
            else:
                action = paddles[i].move(i, paddles, balls, table_size, curr_states, False, 1.0)
        
        
        
        for ball in balls:
            inv_move_factor = int((ball.speed[0]**2+ball.speed[1]**2)**.5)
            if inv_move_factor > 0:
                for i in range(inv_move_factor):
                    ball.move(paddles, table_size, 1./inv_move_factor)
            else:
                ball.move(paddles, table_size, 1)
        
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
        balls, score, idxs = check_point(score, balls, table_size)
        
        
        curr_rewards = []
        if score != old_score:
            if score[0] != old_score[0]:
                #-1 for each point opponent scores
                curr_rewards.append(-1)
                curr_rewards.append(-1)
            else:
                #+1 each time our team scores
                curr_rewards.append(1)
                curr_rewards.append(1)
        else:
            # Reward 0 if nothing happens?
            curr_rewards.append(0)
            curr_rewards.append(0)
            
        
        if (np.random.random() < eps) or score != old_score:
            states.append(curr_states)
            actions.append(curr_actions)
            next_states.append(new_states)
            rewards.append(curr_rewards)
        


################       SCREEN RENDER       ########################

        if yesRender:
            render(screen, paddles, balls, score, table_size)

##########################################################################



    print(score)
    print("states: ", len(states), "actions: ", len(actions), "rewards: ", len(rewards), "next_states: ", len(next_states))
    
    return states, actions, rewards, next_states


def init_game():
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
            return foosPong_ai(states, id)
        else:
            return pong_ai(paddle_frect, ball_frect, table_size)
    
    
    # Set move getter functions
    paddles[0].move_getter = move_getter
    paddles[1].move_getter = move_getter
    paddles[2].move_getter = move_getter
    paddles[3].move_getter = move_getter
            
    eps = 1.0
    yesRender = True
    withTFmodel = True
    
    if withTFmodel:
        foosPong = foosPong_model()
        foosPong.load_weights('./trained_weights/foosPong_model_integrated')
       
    #memories =
    episodes = 100
    memory_states = []
    memory_actions = []
    memory_rewards = []
    memory_next_states = []
    for ep in range(episodes):
        print(f"\nEpisode: {ep}")
        ep_states, ep_actions, ep_rewards, ep_next_states = game_loop(screen, paddles, balls, table_size, clock_rate, turn_wait_rate, score_to_win, 1, eps=eps, yesRender=yesRender, withTFmodel=withTFmodel)
        memory_states = memory_states + ep_states
        memory_actions = memory_actions + ep_actions
        memory_rewards = memory_rewards + ep_rewards
        memory_next_states = memory_next_states + ep_next_states
        
        print("memory_states: ", len(memory_states), "memory_actions: ", len(memory_actions), "memory_rewards: ", len(memory_rewards), "memory_next_states: ", len(memory_next_states))
        
    with open("memory_states.txt", "wb") as fp:
        pickle.dump(memory_states, fp)
    print("States dumped...")
    
    with open("memory_actions.txt", "wb") as fp:
        pickle.dump(memory_actions, fp)
    print("Actions dumped...")
    
    with open("memory_rewards.txt", "wb") as fp:
        pickle.dump(memory_rewards, fp)
    print("Rewards dumped...")
    
    with open("memory_next_states.txt", "wb") as fp:
        pickle.dump(memory_next_states, fp)
    print("Next_states dumped...")
    
    print(np.asarray(memory_states).shape)
    
    pygame.quit()


if __name__ == '__main__':
    pygame.init()
    init_game()

 #   PongAIvAI
#   Authors: Michael Guerzhoy and Denis Begun, 2014-2016.
#   http://www.cs.toronto.edu/~guerzhoy/
#   Email: guerzhoy at cs.toronto.edu
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version. You must credit the authors
#   for the original parts of this code.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   Parts of the code are based on T. S. Hayden Dennison's PongClone (2011)
#   http://www.pygame.org/project-PongClone-1740-3032.html

#   This code runs with Python 2 and requires PyGame for Python 2
#   Download PyGame here: https://bitbucket.org/pygame/pygame/downloads


import pygame, sys, time, random, os
from pygame.locals import *

import math
import numpy as np
import pickle


white = [255, 255, 255]
black = [0, 0, 0]
clock = pygame.time.Clock()

class fRect:
    """Like PyGame's Rect class, but with floating point coordinates"""

    def __init__(self, pos, size):
        self.pos = (pos[0], pos[1])
        self.size = (size[0], size[1])
    def move(self, x, y):
        return fRect((self.pos[0]+x, self.pos[1]+y), self.size)

    def move_ip(self, x, y, move_factor = 1):
        self.pos = (self.pos[0] + x*move_factor, self.pos[1] + y*move_factor)

    def get_rect(self):
        return Rect(self.pos, self.size)

    def copy(self):
        return fRect(self.pos, self.size)

    def intersect(self, other_frect):
        # two rectangles intersect iff both x and y projections intersect
        for i in range(2):
            if self.pos[i] < other_frect.pos[i]: # projection of self begins to the left
                if other_frect.pos[i] >= self.pos[i] + self.size[i]:
                    return 0
            elif self.pos[i] > other_frect.pos[i]:
                if self.pos[i] >= other_frect.pos[i] + other_frect.size[i]:
                    return 0
        return 1 #self.size > 0 and other_frect.size > 0


class Paddle:
    def __init__(self, pos, size, speed, max_angle,  facing, timeout):
        self.frect = fRect((pos[0]-size[0]/2, pos[1]-size[1]/2), size)
        self.speed = speed
        self.size = size
        self.facing = facing
        self.max_angle = max_angle
        self.timeout = timeout

    def factor_accelerate(self, factor):
        self.speed = factor*self.speed

    def move(self, i, paddles, balls, table_size):
        
        # The program crashes if move_getter crashes. The runtime of 
        # move_getter is not limited
        
        
        
        
        ######################################## Alter this to input all balls and all people positions
        ## This is the meat and grits portion that needs changing. This is basically the "predict" step
        ## Need to train the policy function that this step uses as "move_getter"
        
        
        closest_distance = 10000
        closest_ball = None
        for ball in balls:
        # Checks distance to each ball
            if np.linalg.norm(np.asarray(ball.get_center()) - np.asarray(self.frect.pos)) < closest_distance:
                closest_distance = np.linalg.norm(np.asarray(ball.get_center()) - np.asarray(self.frect.pos))
                closest_ball = ball
            
        
        direction = self.move_getter(self.frect.copy(), closest_ball.frect.copy(), tuple(table_size))
        
        ## What its going to look like
        
        #direction = self.move_getter(paddles.copy(), balls.copy(), tuple(table_size))
        
        
        
        
        
        
        
        # The program continues if move_getter crashes. The runtime of
        # move_getter is limited
        # direction = timeout(self.move_getter, (self.frect.copy(), enemy_frect.copy(), ball_frect.copy(), tuple(table_size)), {}, self.timeout)
        
        if direction == "up":
            self.frect.move_ip(0, -self.speed)
        elif direction == "down":
            self.frect.move_ip(0, self.speed)

#        for j in range(len(paddles)):
#            if paddles[j].facing == self.facing and i != j:
#
#                # bottom of current paddle - top of other paddle (on top of other)
#                if ((self.frect.pos[1] + self.frect.size[1]) - (paddles[j].frect.pos[1])) < 0:
#                    self.frect.move_ip(0, ((self.frect.pos[1]+self.frect.size[1]) - (paddles[j].frect.pos[1])))
#
#                # bottom of other paddle - top of current paddle (below other)
#                elif ((paddles[j].frect.pos[1] + paddles[j].frect.size[1]) - self.frect.pos[1]) < 0:
#                    self.frect.move_ip(0, -((paddles[j].frect.pos[1] + paddles[j].frect.size[1]) - self.frect.pos[1]))
#

        to_bottom = (self.frect.pos[1]+self.frect.size[1])-table_size[1]
        if to_bottom > 0:
            self.frect.move_ip(0, -to_bottom)
            
        to_top = self.frect.pos[1]
        if to_top < 0:
            self.frect.move_ip(0, -to_top)
        
        if direction == "up":
            return 1
        else:
            return 0


    def get_face_pts(self):
        return ((self.frect.pos[0] + self.frect.size[0]*self.facing, self.frect.pos[1]),
                (self.frect.pos[0] + self.frect.size[0]*self.facing, self.frect.pos[1] + self.frect.size[1]-1)
                )

    def get_angle(self, y):
        center = self.frect.pos[1]+self.size[1]/2
        rel_dist_from_c = ((y-center)/self.size[1])
        rel_dist_from_c = min(0.5, rel_dist_from_c)
        rel_dist_from_c = max(-0.5, rel_dist_from_c)
        sign = 1-2*self.facing

        return sign*rel_dist_from_c*self.max_angle*math.pi/180





class Ball:
    def __init__(self, table_size, size, paddle_bounce, wall_bounce, dust_error, init_speed_mag):
        rand_ang = (.4+.4*random.random())*math.pi*(1-2*(random.random()>.5))+.5*math.pi
        speed = (init_speed_mag*math.cos(rand_ang), init_speed_mag*math.sin(rand_ang))
        pos = (table_size[0]/2, table_size[1]/2)
        self.frect = fRect((pos[0]-size[0]/2, pos[1]-size[1]/2), size)
        self.speed = speed
        self.size = size
        self.paddle_bounce = paddle_bounce
        self.wall_bounce = wall_bounce
        self.dust_error = dust_error
        self.init_speed_mag = init_speed_mag
        self.prev_bounce = None

    def get_center(self):
        return (self.frect.pos[0] + .5*self.frect.size[0], self.frect.pos[1] + .5*self.frect.size[1])


    def get_speed_mag(self):
        return math.sqrt(self.speed[0]**2+self.speed[1]**2)

    def factor_accelerate(self, factor):
        self.speed = (factor*self.speed[0], factor*self.speed[1])



    def move(self, paddles, table_size, move_factor):
        moved = 0
        walls_Rects = [Rect((-100, -100), (table_size[0]+200, 100)),
                       Rect((-100, table_size[1]), (table_size[0]+200, 100))]

        for wall_rect in walls_Rects:
            if self.frect.get_rect().colliderect(wall_rect):
                c = 0
                
                while self.frect.get_rect().colliderect(wall_rect):
                    self.frect.move_ip(-.1*self.speed[0], -.1*self.speed[1], move_factor)
                    c += 1 # this basically tells us how far the ball has traveled into the wall
                r1 = 1+2*(random.random()-.5)*self.dust_error
                r2 = 1+2*(random.random()-.5)*self.dust_error

                self.speed = (self.wall_bounce*self.speed[0]*r1, -self.wall_bounce*self.speed[1]*r2)
                
                while c > 0 or self.frect.get_rect().colliderect(wall_rect):
                    self.frect.move_ip(.1*self.speed[0], .1*self.speed[1], move_factor)
                    c -= 1 # move by roughly the same amount as the ball had traveled into the wall
                moved = 1
                

        for paddle in paddles:
            if self.frect.intersect(paddle.frect):
                if (paddle.facing == 1 and self.get_center()[0] < paddle.frect.pos[0] + paddle.frect.size[0]/2) or \
                (paddle.facing == 0 and self.get_center()[0] > paddle.frect.pos[0] + paddle.frect.size[0]/2):
                    continue
                
                c = 0
                
                while self.frect.intersect(paddle.frect) and not self.frect.get_rect().colliderect(walls_Rects[0]) and not self.frect.get_rect().colliderect(walls_Rects[1]):
                    self.frect.move_ip(-.1*self.speed[0], -.1*self.speed[1], move_factor)
                    
                    c += 1
                theta = paddle.get_angle(self.frect.pos[1]+.5*self.frect.size[1])
                

                v = self.speed

                v = [math.cos(theta)*v[0]-math.sin(theta)*v[1],
                             math.sin(theta)*v[0]+math.cos(theta)*v[1]]

                v[0] = -v[0]

                v = [math.cos(-theta)*v[0]-math.sin(-theta)*v[1],
                              math.cos(-theta)*v[1]+math.sin(-theta)*v[0]]


                # Bona fide hack: enforce a lower bound on horizontal speed and disallow back reflection
#                if  v[0]*(2*paddle.facing-1) < 1: # ball is not traveling (a) away from paddle (b) at a sufficient speed
#                    v[1] = (v[1]/abs(v[1]))*math.sqrt(v[0]**2 + v[1]**2 - 1) # transform y velocity so as to maintain the speed
#                    v[0] = (2*paddle.facing-1) # note that minimal horiz speed will be lower than we're used to, where it was 0.95 prior to the  increase by 1.2

                #a bit hacky, prevent multiple bounces from accelerating
                #the ball too much
                if not paddle is self.prev_bounce:
                    self.speed = (v[0]*self.paddle_bounce, v[1]*self.paddle_bounce)
                else:
                    self.speed = (v[0], v[1])
                self.prev_bounce = paddle
                

                while c > 0 or self.frect.intersect(paddle.frect):
                
                    self.frect.move_ip(.1*self.speed[0], .1*self.speed[1], move_factor)
                    
                    c -= 1
                
                moved = 1
                

        if not moved:
            self.frect.move_ip(self.speed[0], self.speed[1], move_factor)


def directions_from_input(paddle_rect, ball_rect, table_size):
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        return "up"
    elif keys[pygame.K_DOWN]:
        return "down"
    else:
        return None




def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        print("TIMEOUT")
        return default
    else:
        return it.result

def render(screen, paddles, balls, score, table_size):
    screen.fill(black)

    for paddle in paddles:
        pygame.draw.rect(screen, white, paddle.frect.get_rect())

    for ball in balls:
        pygame.draw.circle(screen, white, (int(ball.get_center()[0]), int(ball.get_center()[1])),  int(ball.frect.size[0]/2), 0)


    pygame.draw.line(screen, white, [screen.get_width()/2, 0], [screen.get_width()/2, screen.get_height()])

    score_font = pygame.font.Font(None, 32)
    screen.blit(score_font.render(str(score[0]), True, white), [int(0.4*table_size[0])-8, 0])
    screen.blit(score_font.render(str(score[1]), True, white), [int(0.6*table_size[0])-8, 0])

    pygame.display.flip()



def check_point(score, balls, table_size):
    for i in range(len(balls)):
        ball = balls[i]
        if ball.frect.pos[0]+ball.size[0]/2 < 0:
            score[1] += 1
            balls[i] = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
            #return (ball, score)
        elif ball.frect.pos[0]+ball.size[0]/2 >= table_size[0]:
            balls[i] = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
            score[0] += 1
            #return (ball, score)

    return (balls, score)


def game_loop(screen, paddles, balls, table_size, clock_rate, turn_wait_rate, score_to_win, display):
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
        states.append(curr_states)
       
        # Take actions...and add to memory actions
        curr_actions = []
        for i in range(len(paddles)):
            action = paddles[i].move(i, paddles, balls, table_size)
            if paddles[i].facing == 0:
                curr_actions.append(action)
        actions.append(curr_actions)
        
        
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
        next_states.append(new_states)
        
        # Check if a ball scored and add rewards accordingly, so rewards[i] should correspond to actions taken at actions[i]
        balls, score = check_point(score, balls, table_size)
        
        curr_rewards = []
        if score != old_score:
            if score[0] != old_score[0]:
                #-1 for each point opponent scores
                curr_rewards.append(-1)
                curr_rewards.append(-1)
            else:
                #rewards big for winning the game, and 2 for each point scored
                if score[1] == score_to_win:
                    curr_rewards.append(10)
                    curr_rewards.append(10)
                else:
                    curr_rewards.append(2)
                    curr_rewards.append(2)
        else:
            # Reward 0 if nothing happens?
            curr_rewards.append(0)
            curr_rewards.append(0)
            
        rewards.append(curr_rewards)
        
#        if not display:
#            continue
##        if score != old_score:
##            font = pygame.font.Font(None, 32)
##            if score[0] != old_score[0]:
##                screen.blit(font.render("Left scores!", True, white, black), [0, 32])
##            else:
##                screen.blit(font.render("Right scores!", True, white, black), [int(table_size[0]/2+20), 32])
##
#
#
#
#
#            pygame.display.flip()
#            clock.tick(turn_wait_rate)


################ REMOVE THIS TO MAKE SIMS GO FAST (SCREEN WON'T RENDER)########################



        #render(screen, paddles, balls, score, table_size)
        
        
        
###############################################################################################

#
#
#
#        pygame.event.pump()
#        keys = pygame.key.get_pressed()
#        if keys[K_q]:
#            return
#
#
#
#        clock.tick(clock_rate)
#
##    font = pygame.font.Font(None, 64)
##    if score[0] > score[1]:
##        screen.blit(font.render("Left wins!", True, white, black), [24, 32])
##    else:
##        screen.blit(font.render("Right wins!", True, white, black), [24, 32])
##    pygame.display.flip()
##    clock.tick(2)
##
##    pygame.event.pump()
##    while any(pygame.key.get_pressed()):
##        pygame.event.pump()
##        clock.tick(30)

    print(score)
    print("states: ", len(states), "actions: ", len(actions), "rewards: ", len(rewards), "next_states: ", len(next_states))
    return states, actions, rewards, next_states


def init_game():
    table_size = (800, 800)
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

    paddles = [Paddle((30, table_size[1]/4), paddle_size, paddle_speed, max_angle,  1, timeout), \
               Paddle((300, table_size[1] - table_size[1]/4), paddle_size, paddle_speed, max_angle,  1, timeout), \
               Paddle((table_size[0] - 30, table_size[1]/4), paddle_size, paddle_speed, max_angle,  0, timeout), \
               Paddle((table_size[0] - 300, table_size[1] - table_size[1]/4), paddle_size, paddle_speed, max_angle, 0, timeout)]
               
    #ball = Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)
    balls = [Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag), Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag), Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag), Ball(table_size, ball_size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)]
    
    
    
    def pong_ai(paddle_frect, ball_frect, table_size):
        if paddle_frect.pos[1] + paddle_frect.size[1]/2 < ball_frect.pos[1] + ball_frect.size[1]/2:
           return "down"
        else:
           return  "up"
    
    # To have The Chaser play against your AI engine, 
    # store your code in student_ai.py, import student_ai, 
    # and set paddles[1].move_getter to student_ai.pong_ai
    paddles[0].move_getter = pong_ai
    paddles[1].move_getter = pong_ai
    paddles[2].move_getter = pong_ai
    paddles[3].move_getter = pong_ai
                        
         
    #memories =
    episodes = 50
    memory_states = []
    memory_actions = []
    memory_rewards = []
    memory_next_states = []
    for ep in range(episodes):
        print(f"\nEpisode: {ep}")
        ep_states, ep_actions, ep_rewards, ep_next_states = game_loop(screen, paddles, balls, table_size, clock_rate, turn_wait_rate, score_to_win, 1)
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
    
    pygame.quit()


if __name__ == '__main__':
    pygame.init()
    init_game()

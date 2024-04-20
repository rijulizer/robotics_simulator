import pygame
import numpy as np

from agent import Agent
from sensor import SensorManager
from environment import Environment
from utils import *

# intialize
pygame.init()
# create window
win_length = 1000
win_height = 800
win = pygame.display.set_mode((win_length, win_height))
pygame.display.set_caption("Robotics Simulation")
# Fonts
font = pygame.font.Font(None, 30)

# define environmnet surface
environment_surface = pygame.Surface((win_length, win_height))
environment_surface.fill((255, 255, 255))
# initalize and draw environment
env = Environment(environment_surface)

# define agent
pos_x = 200
pos_y = 200
radius = 30
theta = 0
color = (255, 0, 0)
number_sensor = 12
sensor_length = 50
sensor_manager = SensorManager(pos_x, pos_y, radius, theta, number_sensor, sensor_length, env)
agent = Agent(pos_x, pos_y, radius, theta, color, sensor_manager)

# define variables
delta_t_max = 5
delta_t_curr = delta_t_max
vl = 0
vl_max = 5 #* delta_t
vl_min = - vl_max
vr = 0
vr_max = 5 #* delta_t
vr_min = - vr_max

sim_run = True
while sim_run:
    pygame.time.delay(50)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sim_run = False
        # read movements
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                vl += 1 #delta_t
                vl = min(vl, vl_max)
            elif event.key == pygame.K_s:
                vl -= 1 #delta_t
                vl = max(vl, vl_min)
            elif event.key == pygame.K_o:
                vr += 1 # delta_t
                vr = min(vr, vl_max)
            elif event.key == pygame.K_l:
                vr -= 1 #delta_t
                vr = max(vr, vr_min)

    # gt the initial / old position of agent
    old_agent_pos = agent.get_pos()
    # get and store points on circle for long step collision detection
    old_points_circle = agent.get_points_circle(8)
    # make agent move
    # simple manipulation to handle the theta increases in clock wise scenario  # Theta
    # increases (+ve) in clock wise direction
    agent.move(vr, vl, delta_t_curr) 
    # get and store points on circle after movement for long step collision detection
    new_points_circle = agent.get_points_circle(8)

    # Check for collision and calculate angle of collision
    collision_angles = check_collision(agent, env)
    print(f"trace_collision - {is_trace_collision(old_points_circle, new_points_circle, env.line_list)}")
    if collision_angles:
        agent.set_pos(old_agent_pos)
        agent.move(vr, vl, delta_t_curr, collision_angles)
        if check_collision(agent, env):
            agent.set_pos(old_agent_pos)
            delta_t_curr -= 0.1
            delta_t_curr = max(delta_t_curr, 0.1)
            continue
        else:
            delta_t_curr = delta_t_max
    print(f"trace_collision - {is_trace_collision(old_points_circle, new_points_circle, env.line_list)}")
    # if the agnts's trace detects collision then dont move the agent, applicable for big delta_ts
    if is_trace_collision(old_points_circle, new_points_circle, env.line_list):
        
        # dont move agent = set to previous position
        agent.set_pos(old_agent_pos)
        # decrease step size and make sure step size never becomes less than 0.1
        delta_t_curr -= 0.1
        delta_t_curr = max(delta_t_curr, 0.1)
        


    # fill the window with white
    win.fill((255, 255, 255))
    # blit the pre rendered environment onto the screen
    win.blit(environment_surface, (0, 0))
    # debug info
    text_vl = font.render(f"v_l: {vl}", True, (0, 0, 0))
    text_vr = font.render(f"v_r: {vr}", True, (0, 0, 0))
    # text_theta = font.render(f"theta: {agent.theta}", True, (0,0,0))
    win.blit(text_vl, (50, 10))
    win.blit(text_vr, (50, 40))
    # win.blit(text_theta, (50, 70))
    pygame.draw.circle(environment_surface, (0, 0, 0), (agent.pos_x, agent.pos_y), 2)
    agent.draw(win)

    # make the agent move based on the vl and vr
    # agent.move(vr, vl, delta_t) # simple manipulation to handle the theta increases in clock wise scenario  # Theta increases (+ve) in clock wise direction
    # draw the agent
    agent.draw(win)
    sensor_manager.draw(win)

    # # refresh the window
    pygame.display.update()
pygame.quit()

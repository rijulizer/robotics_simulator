import pygame
import numpy as np

from agent import Agent
# intialize
pygame.init()
# create window
win_length = 1000
win_height = 800
win = pygame.display.set_mode((win_length,win_height))
pygame.display.set_caption("Robotics Simulation")
# Fonts
font = pygame.font.Font(None, 30)

def draw_env():
    # define environment
    border_x = 100
    border_y = 100
    border_len = 800
    border_height = 600
    border_width = 5
    # define obstacle line-1
    line_1_start_pos_x = border_x + int(border_len/4)
    line_1_end_pos_x = border_x + int(border_len/4)
    line_1_start_pos_y = border_y # starts from the top
    line_1_end_pos_y = int((border_y + border_height) - border_height/3)
    # define obstacle line-1
    line_2_start_pos_x = border_x + 2 * int(border_len/4)
    line_2_end_pos_x = border_x + 2 * int(border_len/4)
    line_2_start_pos_y = border_y + border_height
    line_2_end_pos_y = border_y + int(border_height/3)
    # define obstacle line-1
    line_3_start_pos_x = border_x + 3 * int(border_len/4)
    line_3_end_pos_x = border_x + 3 * int(border_len/4)
    line_3_start_pos_y = border_y # starts from the top
    line_3_end_pos_y = int((border_y + border_height) - border_height/3)
    # define the rectangular border
    pygame.draw.rect(win, (0,0,0), (border_x, border_y, border_len, border_height), width=border_width)
    # draw obscale lines
    pygame.draw.line(win, (0,0,0), (line_1_start_pos_x, line_1_start_pos_y), (line_1_end_pos_x, line_1_end_pos_y))
    pygame.draw.line(win, (0,0,0), (line_2_start_pos_x, line_2_start_pos_y), (line_2_end_pos_x, line_2_end_pos_y))
    pygame.draw.line(win, (0,0,0), (line_3_start_pos_x, line_3_start_pos_y), (line_3_end_pos_x, line_3_end_pos_y))
            
# define agent
pos_x = 200
pos_y = 200
radius = 30
theta = 0
color = (255,0,0)
agent = Agent(pos_x, pos_y , radius, theta, color)

delta_t = 1
vl = 0
vl_max = 5 *  delta_t
vl_min = - vl_max
vr = 0
vr_max = 5 *  delta_t
vr_min = - vr_max

sim_run = True
while sim_run:
    pygame.time.delay(50)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sim_run = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                vl += delta_t
                vl = min(vl, vl_max)
            elif event.key == pygame.K_s:
                vl -= delta_t
                vl = max(vl, vl_min)
            elif event.key == pygame.K_o:
                vr += delta_t
                vr = min(vr, vl_max)
            elif event.key == pygame.K_l:
                vr -= delta_t
                vr = max(vr, vr_min)


    # fill the window with black # debug info
    win.fill((255,255,255))
    text_vl = font.render(f"v_l: {vl}", True, (0,0,0))
    text_vr = font.render(f"v_r: {vr}", True, (0,0,0))
    # text_theta = font.render(f"theta: {agent.theta}", True, (0,0,0))
    win.blit(text_vl, (50, 10))
    win.blit(text_vr, (50, 40))
    # win.blit(text_theta, (50, 70))
    
    
    # make the agent move based on the vl and vr
    agent.move(vr, vl, delta_t) # simple manipulation to handle the theta increases in clock wise scenario  # Theta increases (+ve) in clock wise direction
    # draw the agent
    agent.draw(win)
    # draw environment
    draw_env()

    
    # refresh the window
    pygame.display.update()
pygame.quit()


import pygame
import numpy as np

from agent import Agent
# intialize
pygame.init()
# create window
win = pygame.display.set_mode((1600,1000))
pygame.display.set_caption("Robotics Simulation")

# define agent
pos_x = 200
pos_y = 200
radius = 30
theta = 0
color = (255,0,0)
agent = Agent(pos_x, pos_y , radius, theta, color)



# Fonts
font = pygame.font.Font(None, 36)
vl = 0
vr = 0
delta_t = 1
sim_run = True
while sim_run:
    pygame.time.delay(50)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sim_run = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                vl += delta_t
            elif event.key == pygame.K_s:
                vl -= delta_t

            elif event.key == pygame.K_o:
                vr += delta_t
            elif event.key == pygame.K_l:
                vr -= delta_t
    # fill the window with black # debug info
    win.fill((255,255,255))
    text_vl = font.render(f"vl: {vl}", True, (0,0,0))
    text_vr = font.render(f"vr: {vr}", True, (0,0,0))
    text_theta = font.render(f"theta: {agent.theta}", True, (0,0,0))
    win.blit(text_vl, (50, 50))
    win.blit(text_vr, (50, 100))
    win.blit(text_theta, (50, 150))
    
    
    # make the agent move based on the vl and vr
    agent.move(vr, vl, delta_t) # simple manipulation to handle the theta increases in clock wise scenario  # Theta increases (+ve) in clock wise direction
    # draw the agent
    agent.draw(win)

    # refresh the window
    pygame.display.update()
pygame.quit()


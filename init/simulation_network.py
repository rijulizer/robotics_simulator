from init.utils.utils import draw_all_network, draw_all
from src.utils.engine import simulate
import torch
import pygame
import numpy as np
from src.agent.network import NetworkFromWeights
from src.GUI.dust import Dust

MAX_VELOCITY = 5.0


def run_network_simulation(
        delta_t: float,
        max_time_steps: int = 2000,
        network: NetworkFromWeights = None,
        agent=None,
        win=None,
        environment_surface=None,
        env=None,
        font=None,
        display=False
):
    # initialize
    initial_dust_q = len(env.dust.group)

    # Define variables
    delta_t_max = delta_t
    delta_t_curr = delta_t_max

    sim_run = True
    freeze = False
    time_step = 1
    model_op = None


    curr_dust_num = initial_dust_q
    count = 0
    fitness = 0

    while sim_run and time_step < max_time_steps:

        # if time_step % 40 == 0:
        #     count += 1
        #     if curr_dust_num > len(env.dust.group):
        #         curr_dust_num = len(env.dust.group)
        #     else:
        #         break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim_run = False

        # get the output from the network
        sensor_data = np.array([s.sensor_text for s in agent.sensor_manager.sensors], dtype=np.float32).reshape(1, -1)

        sensor_input = np.array([sensor / agent.sensor_manager.sensor_length for sensor in sensor_data])

        model_op = network.forward(torch.from_numpy(sensor_input), model_op)
        # extarct wheel speed from the model output
        vl = model_op.detach().numpy()[0][0]
        vr = model_op.detach().numpy()[0][1]

        # core logic starts here
        success, collide = simulate(agent,
                                    vr,
                                    vl,
                                    delta_t_curr,
                                    env.line_list,
                                    env.landmarks,
                                    time_step,
                                    False
                                    )

        if collide:
            fitness = -0.5
            break

        if not success:
            delta_t_curr -= 0.1
        else:
            delta_t_curr = delta_t_max

        # update the time ste
        time_step += 1

        if display:
            draw_all(win, environment_surface, agent, vl, vr, delta_t, freeze, time_step, font, env)
        else:
            draw_all_network(win, agent, env)

    dust_collect = np.round((initial_dust_q - len(env.dust.group)), 3)

    if dust_collect < 5:
        dust_collect = 0.00001

    # Find the closest dot to the agent
    distance_min = 100000
    for dot in env.dust.group:
        distance = np.sqrt((dot.pos_x - agent.pos_x) ** 2 + (dot.pos_x - agent.pos_y) ** 2)
        if distance < distance_min:
            distance_min = distance

    if distance_min == 0:
        distance_min = 1

    fitness += dust_collect + (1 / distance_min)

    # Warm Up Chromosomes
    # fitness *= count

    pygame.quit()
    return fitness

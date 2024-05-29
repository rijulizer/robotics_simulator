from init.utils.utils import draw_all
from src.utils.engine import simulate
import torch
import pygame
import numpy as np
from src.agent.network import NetworkFromWeights


def run_network_simulation(
        delta_t: float,
        max_time_steps: int = 2000,
        network: NetworkFromWeights = None,
        agent=None,
        win=None,
        environment_surface=None,
        env=None,
        font=None
):
    # initialize
    initial_dust_q = len(env.dust.group)

    # Define variables
    delta_t_max = delta_t
    delta_t_curr = delta_t_max

    sim_run = True
    freeze = False
    time_step = 0
    # initialize delyaed model output
    model_op = None
    while sim_run and time_step < max_time_steps:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim_run = False
        
        # get the output from the network
        sensor_data = np.array([s.sensor_text for s in agent.sensor_manager.sensors], dtype=np.float32).reshape(1, -1)
        model_op = network.forward(torch.from_numpy(sensor_data), model_op)
        # extarct wheel speed from the model output
        vl = model_op.detach().numpy()[0][0]
        vr = model_op.detach().numpy()[0][1]

        # core logic starts here
        success = simulate(agent,
                           vr,
                           vl,
                           delta_t_curr,
                           env.line_list,
                           env.landmarks,
                           time_step,
                           False
                           )

        if not success:
            delta_t_curr -= 0.1
        else:
            delta_t_curr = delta_t_max

        # update the time ste
        time_step += 1

        draw_all(win, environment_surface, agent, vl, vr, delta_t, freeze, time_step, font, env)

    final_dust_q = len(env.dust.group)
    dust_remains = np.round(((initial_dust_q - final_dust_q) / initial_dust_q), 2)
    pygame.quit()
    return dust_remains

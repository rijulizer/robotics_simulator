from init.utils.utils import draw_all
from src.utils.engine import simulate
import torch
import pygame
import numpy as np
from src.agent.network import NetworkFromWeights

MAX_VELOCITY = 5.0

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
    fitness_param_dust_initial = len(env.dust.group)

    # Define variables
    delta_t_max = delta_t
    delta_t_curr = delta_t_max

    sim_run = True
    freeze = False
    time_step = 0
    # initialize delyaed model output
    model_op = None
    fitness_param_velchange = 0.0
    max_velocity_change = 10
    agent_track = []
    agent_energy = []
    fitness_param_collision = []
    while sim_run and time_step < max_time_steps:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim_run = False

        # get the output from the network
        sensor_data = np.array([s.sensor_text for s in agent.sensor_manager.sensors], dtype=np.float32).reshape(1, -1)
        fitness_param_collision.append(np.min(sensor_data))
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
        # track agents path
        agent_track.append((round(agent.pos_x,1), round(agent.pos_y,1)))
        # track agent's normalised energy use
        agent_energy.append((abs(vl) + abs(vr)) / 2 * MAX_VELOCITY)

        if not success:
            delta_t_curr -= 0.1
        else:
            delta_t_curr = delta_t_max

        # update the time ste
        time_step += 1
        vel_change += abs((vl + vr) / max_velocity_change)

        draw_all(win, environment_surface, agent, vl, vr, delta_t, freeze, time_step, font, env)
    # return entities for fitness function 
    final_dust_q = len(env.dust.group)
    dust_collect = np.round(((initial_dust_q - final_dust_q) / initial_dust_q), 3)
    unique_positions = np.round(len(set(agent_track)) / max_time_steps, 3)
    energy_used = np.round(sum(agent_energy) / max_time_steps, 3)
    vel_change = np.round((vel_change/time_step),2)
    fitness_param_collision = np.array(fitness_param_collision)
    fitness_param_collision = np.round(((np.abs(fitness_param_collision)<1.0).sum()/fitness_param_collision.size),2)
    


    pygame.quit()
    return dust_collect, unique_positions, energy_used, fitness_param_velchange, fitness_param_collision


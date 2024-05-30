import pygame
from src.agent.agent import Agent
from src.agent.sensor import SensorManager
from src.environment import Environment
from src.engine import simulate
from src.utils import draw_belief_ellipse
import pickle as pkl
import numpy as np
import logging

# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')

GLOBAL_SCALE = 50
WIN_LENGTH = 1200
WIN_HEIGHT = 800
ENV_OFFSET_X = 20
ENV_OFFSET_Y = 200

def draw_all(win, environment_surface, agents, velocities, delta_t, freeze, time_step, font):
    # Fill the window with white
    win.fill((226, 200, 159 ))

    # Blit the pre rendered environment onto the screen
    win.blit(environment_surface, (0, 0))
    
    vl ,vr = velocities[0]
    vl_2, vr_2 = velocities[1]

    # Wheel velocity display
    text_v_1 = font.render(f"vel_1: ||{vl}|| --- ||{vr}||", True, (0, 0, 0))
    text_v_2 = font.render(f"vel_2: ||{vl_2}|| --- ||{vr_2}||", True, (0, 0, 0))
    text_delta_t = font.render(f"delta_t: {delta_t}", True, (0, 0, 0))
    text_freeze = font.render(f"freeze: {freeze}", True, (0, 0, 0))
    win.blit(text_v_1, (50, 10))
    win.blit(text_v_2, (300, 10))
    win.blit(text_delta_t, (50, 70))
    win.blit(text_freeze, (200, 70))


    # get the agents belief
    # agent.get_belief(detected_landmarks)
    # logging.debug(f"Actual position: {agent.pos_x, agent.pos_y, agent.theta}")
    # logging.debug(f"Belief: {agent.bel_pos_x, agent.bel_pos_y, agent.bel_theta}")

    # Agent trajectory
    # pygame.draw.circle(environment_surface, (34, 139, 34), (agent.pos_x, agent.pos_y), 2)
    # pygame.draw.circle(environment_surface, (240, 90, 90), (agent.bel_pos_x, agent.bel_pos_y), 2)
    # pygame.draw.circle(environment_surface, (0, 0, 70), (agent.est_bel_pos_x, agent.est_bel_pos_y), 1)
    # print the belief cov matrix in every 100th iteration
    # if time_step % 100 == 0:
    #     draw_belief_ellipse(environment_surface, agent.bel_cov, agent.bel_pos_x, agent.bel_pos_y, scale= 0.5 * GLOBAL_SCALE)
    for agent in agents:
        if time_step % 50 == 0:
            agent.create_map(environment_surface)

        # detect landmarks
        # detected_landmarks = agent.sensor_manager.scan_landmarks(env.landmarks)
        # for i in detected_landmarks:
        #     landmark_phi = font.render(f"{i[0], i[1], round(i[4]* 180 / 3.14,2)}", True, (0, 0, 0))
        #     win.blit(landmark_phi, (i[0], i[1]))
        logging.debug(f"agent_bel_cov: {agent.bel_cov}")
        agent.draw(win)  # Draw agent components

    # Refresh the window
    pygame.display.update()


def _init_GUI(num_landmarks):
    # initialize
    pygame.init()
    # Create window
    win_length = WIN_LENGTH
    win_height = WIN_HEIGHT
    win = pygame.display.set_mode((win_length, win_height))
    pygame.display.set_caption("Robotics Simulation")
    # Fonts
    font = pygame.font.Font(None, 30)

    # Define environment surface
    environment_surface = pygame.Surface((win_length, win_height))
    environment_surface.fill((226, 200, 159 ))
    # Initialize and draw environment
    env = Environment(environment_surface)
    # put landmarks on the environment
    env.put_landmarks(environment_surface, num_landmarks)
    # define agent
    pos_x_1 = ENV_OFFSET_X + 1 * GLOBAL_SCALE
    pos_y_1 = ENV_OFFSET_Y + 1 * GLOBAL_SCALE
    pos_x_2 = ENV_OFFSET_X + 8 * GLOBAL_SCALE
    pos_y_2 = ENV_OFFSET_Y + 5 * GLOBAL_SCALE
    radius = 0.3 * GLOBAL_SCALE
    theta = 0
    number_sensors = 12
    sensor_length = 1.5 * GLOBAL_SCALE
    agents = []
    
    agent_1 = Agent(pos_x_1, pos_y_1, radius, theta, filter=False, color=(50, 150, 230))
    agent_2 = Agent(pos_x_2, pos_y_2, radius, theta, filter=False, color=(255, 35, 35))
    
    sensor_manager_1 = SensorManager(agent_1.get_agent_stats(),
                                   number_sensors,
                                   sensor_length,
                                   env.line_list)
    
    sensor_manager_2 = SensorManager(agent_2.get_agent_stats(),
                                   number_sensors,
                                   sensor_length,
                                   env.line_list)
    # convert radians to degrees
    theta = theta * 180 / 3.14
    # attach sensors to agent
    agent_1.set_sensor(sensor_manager_1)
    agent_2.set_sensor(sensor_manager_2)
    agents = [agent_1, agent_2]
    # detect landmarks
    # detected_landmarks = agent.sensor_manager.scan_landmarks(env.landmarks)
    # win.blit(environment_surface, (0, 0))
    # logging.debug(f"Detected Landmarks: {detected_landmarks}")

    return win, environment_surface, agents, font, env

def run_simulation(
        delta_t: float,
        track: bool = False,
        num_landmarks: int = 20
):
    # initialize
    win, environment_surface, agents, font, env = _init_GUI(num_landmarks)

    # Define variables
    delta_t_max = delta_t
    delta_t_curr = delta_t_max
    vl_max = 5  # * delta_t
    vl_min = - vl_max
    vr_max = 5  # * delta_t
    vr_min = - vr_max
    vl = 0
    vr = 0
    vl_2 = 0
    vr_2 = 0

    sim_run = True
    freeze = False
    time_step = 0
    tracker = []
    delta_x, delta_y, delta_theta, vl_list, vr_list = [], [], [], [], []
    while sim_run:
        pygame.time.delay(25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim_run = False
            # read movements
            if event.type == pygame.KEYDOWN:
                # 2 wheels drive
                # if event.key == pygame.K_w and not freeze:
                #     vl += 1  # delta_t
                #     vl = min(vl, vl_max)
                # elif event.key == pygame.K_s and not freeze:
                #     vl -= 1  # delta_t
                #     vl = max(vl, vl_min)
                # elif event.key == pygame.K_o and not freeze:
                #     vr += 1  # delta_t
                #     vr = min(vr, vr_max)
                # elif event.key == pygame.K_l and not freeze:
                #     vr -= 1  # delta_t
                #     vr = max(vr, vr_min)
                # elif event.key == pygame.K_SPACE:
                #     freeze = not freeze
                # up arrow key
                # normal drive for first agent
                if event.key == pygame.K_w and not freeze:
                    vl = min(vl + 1, vl_max)
                    vr = min(vr+1, vr_max)
                elif event.key == pygame.K_s and not freeze:
                    vl = max(vl - 1, vl_min)
                    vr = max(vr - 1, vr_min)
                elif event.key == pygame.K_a and not freeze:
                    vl = max(vl - 1, vl_min)
                    vr = min(vr + 1, vr_max)
                elif event.key == pygame.K_d and not freeze:
                    vl = min(vl + 1, vl_max)
                    vr = max(vr - 1, vr_min)
                
                # normal drive for Second agent
                elif event.key == pygame.K_UP and not freeze:
                    vl_2 = min(vl_2 + 1, vl_max)
                    vr_2 = min(vr_2 + 1, vr_max)
                elif event.key == pygame.K_DOWN and not freeze:
                    vl_2 = max(vl_2 - 1, vl_min)
                    vr_2 = max(vr_2 - 1, vr_min)
                elif event.key == pygame.K_LEFT and not freeze:
                    vl_2 = max(vl_2 - 1, vl_min)
                    vr_2 = min(vr_2 + 1, vr_max)
                elif event.key == pygame.K_RIGHT and not freeze:
                    vl_2 = min(vl_2 + 1, vl_max)
                    vr_2 = max(vr_2 - 1, vr_min)
                elif event.key == pygame.K_SPACE:
                    freeze = not freeze

                
        # core logic starts here
        if not freeze:
            success = simulate(agents[0],
                               vr,
                               vl,
                               delta_t_curr,
                               env.line_list,
                               env.landmarks,
                               time_step
                               )
            success_2 = simulate(agents[1],
                               vr_2,
                               vl_2,
                               delta_t_curr,
                               env.line_list,
                               env.landmarks,
                               time_step
                               )

            if not (success and success_2):
                delta_t_curr -= 0.1
            else:
                delta_t_curr = delta_t_max

            # update the time ste
            time_step += 1

            if track:
                tracker.append((vl, vr))

        velocities = ((vl, vr), (vl_2, vr_2))  
        draw_all(win, environment_surface, agents, velocities, delta_t, freeze, time_step, font)

    # save the tracker
    if track:
        with open("tracker.pkl", "wb") as f:
            pkl.dump(tracker, f)

    pygame.quit()



if __name__ == "__main__":
    save_s = False  # False or True (True - Run the simulation with saved trajectory data)
    plot_graph = False  # False or True (True - Plot the graph)
    track_res = False  # False or True (True - Save the trajectory data)
    graph_plot = None
    # Start Timer for the simulation with python in build function
    run_simulation(
        delta_t=1,
        track=track_res,
        num_landmarks=10,
        )
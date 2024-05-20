import pygame
from src.agent.agent import Agent
from src.agent.sensor import SensorManager
from src.environment import Environment
from src.engine import simulate
from src.utils import draw_belief_ellipse
from src.graphGUI.graphs import GraphGUI
from src.inetwork import INetwork
import pickle as pkl
import numpy as np
import logging

# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')


def draw_all(win, environment_surface, agent, vl, vr, delta_t, freeze, time_step, font,env):
    # Fill the window with white
    win.fill((255, 255, 255))

    # Blit the pre rendered environment onto the screen
    win.blit(environment_surface, (0, 0))

    # Wheel velocity display
    text_vl = font.render(f"v_l: {vl}", True, (0, 0, 0))
    text_vr = font.render(f"v_r: {vr}", True, (0, 0, 0))
    text_delta_t = font.render(f"delta_t: {delta_t}", True, (0, 0, 0))
    text_freeze = font.render(f"freeze: {freeze}", True, (0, 0, 0))
    win.blit(text_vl, (50, 10))
    win.blit(text_vr, (50, 40))
    win.blit(text_delta_t, (50, 70))
    win.blit(text_freeze, (200, 10))
    

    # get the agents belief
    # agent.get_belief(detected_landmarks)
    # logging.debug(f"Actual position: {agent.pos_x, agent.pos_y, agent.theta}")
    # logging.debug(f"Belief: {agent.bel_pos_x, agent.bel_pos_y, agent.bel_theta}")

    # Agent trajectory
    pygame.draw.circle(environment_surface, (34, 139, 34), (agent.pos_x, agent.pos_y), 2)
    #pygame.draw.circle(environment_surface, (240, 90, 90), (agent.bel_pos_x, agent.bel_pos_y), 2)
    #pygame.draw.circle(environment_surface, (0, 0, 70), (agent.est_bel_pos_x, agent.est_bel_pos_y), 1)
    # print the belief cov matrix in every 100th iteration
    # if time_step % 100 == 0:
    #     draw_belief_ellipse(environment_surface, agent.bel_cov, agent.bel_pos_x, agent.bel_pos_y, scale=100)

    # detect landmarks
    # detected_landmarks = agent.sensor_manager.scan_landmarks(env.landmarks)
    # for i in detected_landmarks:
    #     landmark_phi = font.render(f"{i[0], i[1], round(i[4]* 180 / 3.14,2)}", True, (0, 0, 0))
    #     win.blit(landmark_phi, (i[0], i[1]))
    logging.debug(f"agent_bel_cov: {agent.bel_cov}")
    agent.draw(win)  # Draw agent components
    env.dust.update(win,agent)

    # Refresh the window
    pygame.display.update()


def _init_GUI(num_landmarks):
    # initialize
    pygame.init()

    # Create window
    win_length = 1000
    win_height = 800
    win = pygame.display.set_mode((win_length, win_height), flags=pygame.HIDDEN)
    pygame.display.set_caption("Robotics Simulation")
    # Fonts
    font = pygame.font.Font(None, 30)

    # Define environment surface
    environment_surface = pygame.Surface((win_length, win_height))
    environment_surface.fill((255, 255, 255))
    # Initialize and draw environment
    env = Environment(environment_surface)
    # put landmarks on the environment
    env.put_landmarks(environment_surface, num_landmarks)
    # define agent
    pos_x = 200
    pos_y = 200
    radius = 30
    theta = 0

    number_sensors = 12
    sensor_length = 10

    agent = Agent(pos_x, pos_y, radius, theta)

    sensor_manager = SensorManager(agent.get_agent_stats(),
                                   number_sensors,
                                   sensor_length,
                                   env.line_list)
    # convert radians to degrees
    theta = theta * 180 / 3.14
    # attach sensors to agent
    agent.set_sensor(sensor_manager)
    # detect landmarks
    # detected_landmarks = agent.sensor_manager.scan_landmarks(env.landmarks)
    # win.blit(environment_surface, (0, 0))
    # logging.debug(f"Detected Landmarks: {detected_landmarks}")

    return win, environment_surface, agent, font, env, sensor_manager


def run_saved_simulation(
        delta_t: float,
        graphGUI: GraphGUI,
        track: list,
        num_landmarks: int = 20,
        file_name_win: str = "Experiment"
):
    if track is None or len(track) == 0:
        raise ValueError("No track data found")

    # initialize
    win, environment_surface, agent, font, env, _ = _init_GUI(num_landmarks)

    # Define variables
    delta_t_max = delta_t
    delta_t_curr = delta_t_max

    freeze = False
    time_step = 0
    delta_x, delta_y, delta_theta, vl_list, vr_list = [], [], [], [], []
    for vl, vr in track:
        pygame.time.delay(1)

        flag = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    freeze = not freeze

        if flag:
            break

        while freeze:
            pygame.time.delay(25)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        freeze = not freeze

        # core logic starts here
        success = simulate(agent,
                           vr,
                           vl,
                           delta_t_curr,
                           env.line_list,
                           env.landmarks,
                           time_step
                           )

        if not success:
            delta_t_curr -= 0.1
        else:
            delta_t_curr = delta_t_max

        # update the time ste
        time_step += 1

        # update the graph
        if graphGUI:
            delta_x.append(abs(agent.pos_x - agent.bel_pos_x))
            delta_y.append(abs(agent.pos_y - agent.bel_pos_y))
            delta_theta.append(abs(agent.theta - agent.bel_theta))
            vl_list.append(vl)
            vr_list.append(vr)
            if time_step % 10 == 0:
                graphGUI.update_plot({"vl": vl_list,
                                      "vr": vr_list,
                                      "delta_x": delta_x,
                                      "delta_y": delta_y,
                                      "delta_theta": delta_theta})
                delta_x, delta_y, delta_theta, vl_list, vr_list = [], [], [], [], []

        draw_all(win, environment_surface, agent, vl, vr, delta_t, freeze, time_step, font,env)

    pygame.image.save(win, "./src/experiments_data/" + file_name_win + ".png")
    pygame.quit()


def run_simulation(
        delta_t: float,
        graphGUI: GraphGUI,
        track: bool = False,
        num_landmarks: int = 20
):
    # initialize
    win, environment_surface, agent, font, env, _ = _init_GUI(num_landmarks)

    # Define variables
    delta_t_max = delta_t
    delta_t_curr = delta_t_max
    vl = 0
    vl_max = 5  # * delta_t
    vl_min = - vl_max
    vr = 0
    vr_max = 5  # * delta_t
    vr_min = - vr_max

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
                if event.key == pygame.K_w and not freeze:
                    vl += 1  # delta_t
                    vl = min(vl, vl_max)
                elif event.key == pygame.K_s and not freeze:
                    vl -= 1  # delta_t
                    vl = max(vl, vl_min)
                elif event.key == pygame.K_o and not freeze:
                    vr += 1  # delta_t
                    vr = min(vr, vr_max)
                elif event.key == pygame.K_l and not freeze:
                    vr -= 1  # delta_t
                    vr = max(vr, vr_min)
                elif event.key == pygame.K_SPACE:
                    freeze = not freeze

        # core logic starts here
        if not freeze:
            success = simulate(agent,
                               vr,
                               vl,
                               delta_t_curr,
                               env.line_list,
                               env.landmarks,
                               time_step
                               )

            if not success:
                delta_t_curr -= 0.1
            else:
                delta_t_curr = delta_t_max

            # update the time ste
            time_step += 1

            if track:
                tracker.append((vl, vr))

            # update the graph
            if graphGUI:
                delta_x.append(abs(agent.pos_x - agent.bel_pos_x))
                delta_y.append(abs(agent.pos_y - agent.bel_pos_y))
                delta_theta.append(abs(agent.theta - agent.bel_theta))
                vl_list.append(vl)
                vr_list.append(vr)
                if time_step % 10 == 0:
                    graphGUI.update_plot({"vl": vl_list,
                                          "vr": vr_list,
                                          "delta_x": delta_x,
                                          "delta_y": delta_y,
                                          "delta_theta": delta_theta})
                    delta_x, delta_y, delta_theta, vl_list, vr_list = [], [], [], [], []

        draw_all(win, environment_surface, agent, vl, vr, delta_t, freeze, time_step, font,env)

    # save the tracker
    if track:
        with open("tracker.pkl", "wb") as f:
            pkl.dump(tracker, f)

    pygame.quit()


def run_network_simulation(
        delta_t: float,
        graphGUI: GraphGUI,
        track: bool = False,
        num_landmarks: int = 20,
        max_time_steps: int = 2000,
        network: object = None
):
    # initialize
    win, environment_surface, agent, font, env, sensormanager = _init_GUI(num_landmarks)
    initial_dust_q = len(env.dust.group.sprites())

    # Define variables
    delta_t_max = delta_t
    delta_t_curr = delta_t_max
    vl = 0
    vl_max = 5  # * delta_t
    vl_min = - vl_max
    vr = 0
    vr_max = 5  # * delta_t
    vr_min = - vr_max
    inet = INetwork(sensormanager.no_sensor,2*vl_max,[vl_min,vr_min],network)

    sim_run = True
    freeze = False
    time_step = 0
    tracker = []
    delta_x, delta_y, delta_theta, vl_list, vr_list = [], [], [], [], []
    while sim_run and max_time_steps >= 0:
        pygame.time.delay(25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim_run = False
            # read movements
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w and not freeze:
                    vl += 1  # delta_t
                    vl = min(vl, vl_max)
                elif event.key == pygame.K_s and not freeze:
                    vl -= 1  # delta_t
                    vl = max(vl, vl_min)
                elif event.key == pygame.K_o and not freeze:
                    vr += 1  # delta_t
                    vr = min(vr, vr_max)
                elif event.key == pygame.K_l and not freeze:
                    vr -= 1  # delta_t
                    vr = max(vr, vr_min)
                elif event.key == pygame.K_SPACE:
                    freeze = not freeze
        sensor_values = [int(s) for s in sensormanager.sensor_read]
        vl,vr = inet.get_input_velocity(sensor_values)
        # core logic starts here
        if not freeze:
            success = simulate(agent,
                               vr,
                               vl,
                               delta_t_curr,
                               env.line_list,
                               env.landmarks,
                               time_step
                               )

            if not success:
                delta_t_curr -= 0.1
            else:
                delta_t_curr = delta_t_max

            # update the time ste
            time_step += 1

            if track:
                tracker.append((vl, vr))

            # update the graph
            if graphGUI:
                delta_x.append(abs(agent.pos_x - agent.bel_pos_x))
                delta_y.append(abs(agent.pos_y - agent.bel_pos_y))
                delta_theta.append(abs(agent.theta - agent.bel_theta))
                vl_list.append(vl)
                vr_list.append(vr)
                if time_step % 10 == 0:
                    graphGUI.update_plot({"vl": vl_list,
                                          "vr": vr_list,
                                          "delta_x": delta_x,
                                          "delta_y": delta_y,
                                          "delta_theta": delta_theta})
                    delta_x, delta_y, delta_theta, vl_list, vr_list = [], [], [], [], []

        draw_all(win, environment_surface, agent, vl, vr, delta_t, freeze, time_step, font, env)
        max_time_steps -=1
    final_dust_q = len(env.dust.group.sprites())
    # save the tracker
    if track:
        with open("tracker.pkl", "wb") as f:
            pkl.dump(tracker, f)

    pygame.quit()
    return initial_dust_q - final_dust_q

def run_experiments(track,
                    num_landmarks=8,
                    file_name_win="Experiment_Draft",
                    exp_name="Experiment_Draft",
                    plot_graph=True):

    if plot_graph:
        graph_plot = GraphGUI()
    else:
        graph_plot = None

    run_saved_simulation(delta_t=1,
                         graphGUI=graph_plot,
                         track=track,
                         num_landmarks=num_landmarks,
                         file_name_win=file_name_win)

    print("Simulation completed successfully")
    # Write in .txt file with name of experiments and the average values of delta x, delta y, delta theta
    if graph_plot and save_s:
        print(len(graph_plot.store1))
        with open(f'./src/experiments_data/{exp_name}_results.txt', 'w') as f:
            f.write(f"Experiment Name: {exp_name}\n")
            f.write(f"Average Delta x: {round(np.mean(graph_plot.store1), 2)}\n")
            f.write(f"Average Delta y: {round(np.mean(graph_plot.store2), 2)}\n")
            f.write(f"Average Delta theta: {round(np.mean(graph_plot.store3), 2)}\n")
        # Save figure plot[
        graph_plot.fig.savefig(f"./src/experiments_data/{exp_name}_graph.png")


save_s = False  # False or True (True - Run the simulation with saved trajectory data)
plot_graph = False  # False or True (True - Plot the graph)
track_res = False  # False or True (True - Save the trajectory data)

if __name__ == "__main__":
    if plot_graph:
        graph_plot = GraphGUI()
    else:
        graph_plot = None
    # Start Timer for the simulation with python in build function
    if not save_s:
        run_simulation(delta_t=1,
                       graphGUI=graph_plot,
                       track=track_res,
                       num_landmarks=0)
    else:
        # Run save simulation
        with open("tracker.pkl", "rb") as f:
            track = pkl.load(f)
        run_experiments(track,
                        file_name_win="Experiment_Draft",
                        exp_name="Experiment_Draft",
                        plot_graph=plot_graph)

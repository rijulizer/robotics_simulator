import pygame
from src.agent.agent import Agent
from src.agent.sensor import SensorManager
from src.environment import Environment
from src.engine import simulate
from src.utils import draw_belief_ellipse
from src.graphGUI.graphs import GraphGUI
import logging


# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')


def run_simulation(
        delta_t: float,
        graphGUI: GraphGUI
):
    # initialize
    pygame.init()

    # Create window
    win_length = 1000
    win_height = 800
    win = pygame.display.set_mode((win_length, win_height))
    pygame.display.set_caption("Robotics Simulation")
    # Fonts
    font = pygame.font.Font(None, 30)

    # Define environment surface
    environment_surface = pygame.Surface((win_length, win_height))
    environment_surface.fill((255, 255, 255))
    # Initialize and draw environment
    env = Environment(environment_surface)
    # put landmarks on the environment
    env.put_landmarks(environment_surface)
    # define agent
    pos_x = 200
    pos_y = 200
    radius = 30
    theta = 0

    number_sensors = 12
    sensor_length = 150

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

            # update the graph
            if graphGUI:
                graphGUI.update_plot({"vl": vl, "vr": vr})

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
        pygame.draw.circle(environment_surface, (240, 90, 90), (agent.bel_pos_x, agent.bel_pos_y), 2)
        pygame.draw.circle(environment_surface, (0, 0, 70), (agent.est_bel_pos_x, agent.est_bel_pos_y), 1)
        # print the belief cov matrix in every 100th iteration
        if time_step % 100 == 0:
            draw_belief_ellipse(environment_surface, agent.bel_cov, agent.bel_pos_x, agent.bel_pos_y, scale=100)

        # detect landmarks
        # detected_landmarks = agent.sensor_manager.scan_landmarks(env.landmarks)
        # for i in detected_landmarks:
        #     landmark_phi = font.render(f"{i[0], i[1], round(i[4]* 180 / 3.14,2)}", True, (0, 0, 0))
        #     win.blit(landmark_phi, (i[0], i[1]))
        logging.debug(f"agent_bel_cov: {agent.bel_cov}")
        agent.draw(win)  # Draw agent components

        time_step += 1
        # Refresh the window
        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    run_simulation(delta_t=1, graphGUI=GraphGUI())
    #run_simulation(delta_t=1)

import pygame
from src.agent.agent import Agent
from src.agent.sensor import SensorManager
from src.GUI.environment import Environment
from src.utils.utils import draw_belief_ellipse


def draw_all(win, environment_surface, agent, vl, vr, delta_t, freeze, time_step, font, env):
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

    # Agent trajectory
    pygame.draw.circle(environment_surface, (34, 139, 34), (agent.pos_x, agent.pos_y), 2)
    # pygame.draw.circle(environment_surface, (240, 90, 90), (agent.bel_pos_x, agent.bel_pos_y), 2)
    # pygame.draw.circle(environment_surface, (0, 0, 70), (agent.est_bel_pos_x, agent.est_bel_pos_y), 1)

    # print the belief cov matrix in every 100th iteration
    if time_step % 100 == 0:
        draw_belief_ellipse(environment_surface, agent.bel_cov, agent.bel_pos_x, agent.bel_pos_y, scale=100)

    agent.draw(win)  # Draw agent components
    env.dust.update(win, agent)

    # Refresh the window
    pygame.display.update()


def draw_all_network(win, agent, env):
    agent.draw(win)  # Draw agent components
    env.dust.update(win, agent)


def _init_GUI(num_landmarks,
              num_sensor,
              sensor_length,
              pygame_flags=None):
    # initialize
    pygame.init()

    # Create window
    win_length = 1000
    win_height = 800
    if pygame_flags:
        win = pygame.display.set_mode((win_length, win_height), flags=pygame_flags)  # pygame.HIDDEN
    else:
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
    env.put_landmarks(environment_surface, num_landmarks)
    # define agent
    pos_x = 150
    pos_y = 150
    radius = 30
    theta = 1.57

    agent = Agent(pos_x, pos_y, radius, theta)

    sensor_manager = SensorManager(agent.get_agent_stats(),
                                   num_sensor,
                                   sensor_length,
                                   env.line_list)

    # attach sensors to agent
    agent.set_sensor(sensor_manager)

    return win, environment_surface, agent, font, env

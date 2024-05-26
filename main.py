from init import simulation_saved, simulation, simulation_network
from init.utils.utils import _init_GUI

delta_t = 1
track = False
num_landmarks = 20
num_sensor = 12
sensor_length = 150


# initialize
win, environment_surface, agent, font, env = _init_GUI(num_landmarks,
                                                       num_sensor,
                                                       sensor_length)

simulation.run_simulation(delta_t,
                          track,
                          agent,
                          win,
                          environment_surface,
                          env,
                          font)
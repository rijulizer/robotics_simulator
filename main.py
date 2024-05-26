from init import simulation_saved, simulation, simulation_network
from src.utils.geneticAlg import GeneticAlgorithm
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
                          "tracker",
                          agent,
                          win,
                          environment_surface,
                          env,
                          font)





########################################################################################################################

#Parameters
# params = {
#     "POP_SIZE": 100,
#     "CHROMOSOME_LENGTH": 10,
#     "GEN_MAX": 100,
#     "SELECTION_PARAM": {
#         "type": "tournament",
#         "num_individuals": 20,
#         "tournament_size": 5
#     },
#     "MUTATION_PARAM": {
#         "type": "uniform",
#         "rate": 0.01
#     },
#     "CROSSOVER_PARAM": {
#         "type": "one_point"
#     }
# }
# genAlg = GeneticAlgorithm(**params)
# print(genAlg)

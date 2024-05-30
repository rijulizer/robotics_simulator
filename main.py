from init import simulation_saved, simulation, simulation_network
from src.utils.geneticAlg import GeneticAlgorithm
# from src.agent.network import NetworkFromWeights
# from init.utils.utils import _init_GUI
import warnings

# Filter out specific UserWarning regarding the use of `.T` on tensors
warnings.filterwarnings("ignore",
                        message="The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release.")

delta_t = 1
track = False
num_landmarks = 0
num_sensor = 12
sensor_length = 150

# initialize
# win, environment_surface, agent, font, env = _init_GUI(num_landmarks,
#                                                        num_sensor,
#                                                        sensor_length)
#
# simulation.run_simulation(delta_t,
#                           track,
#                           "tracker",
#                           agent,
#                           win,
#                           environment_surface,
#                           env,
#                           font)


########################################################################################################################

#Parameters
params = {
    "POP_SIZE": 100,
    "CHROMOSOME_LENGTH": 240,
    "GEN_MAX": 100,
    "SELECTION_PARAM": {
        "type": "tournament",
        "num_individuals": 20,
        "tournament_size": 5
    },
    "MUTATION_PARAM": {
        "type": "uniform",
        "rate": 0.01
    },
    "CROSSOVER_PARAM": {
        "type": "two_point"
    }
}


def main():
    # Your existing code that initializes and starts the genetic algorithm
    genAlg = GeneticAlgorithm(**params)
    genAlg.genetic_algorithm()


if __name__ == '__main__':
    # This line is critical for multiprocessing under Windows
    from multiprocessing import freeze_support

    freeze_support()
    main()

########################################################################################################################

# Run the network simulation

# chromosome = "011101110011101000110011001001100111000000011011011101010001011110110110000101000111100010000100111001001011111111101011010100111111000111010001111101000101100011110010100110001000111010110111010111111111011001110111111010011100111001011001"
#
# network = NetworkFromWeights(chromosome, v_max=10.0)
# # Run simulation
# win, environment_surface, agent, font, env = _init_GUI(num_landmarks,
#                                                        num_sensor,
#                                                        sensor_length)
#
# max_time_steps = 10000000000
# dust_remains = simulation_network.run_network_simulation(delta_t,
#                                                          max_time_steps,
#                                                          network,
#                                                          agent,
#                                                          win,
#                                                          environment_surface,
#                                                          env,
#                                                          font)

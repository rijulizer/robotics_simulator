from init import simulation_saved, simulation, simulation_network
from src.utils.geneticAlg import GeneticAlgorithm
from src.agent.network import NetworkFromWeights, NetworkFromWeights_2
from init.utils.utils import _init_GUI
import warnings


# Filter out specific UserWarning regarding the use of `.T` on tensors
warnings.filterwarnings("ignore",
                        message="The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release.")

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

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
    "POP_SIZE": 200,
    "CHROMOSOME_LENGTH": 54*5,
    "GEN_MAX": 100,
    "SELECTION_PARAM": {
        "type": "roulette",
        "num_individuals": 30,
        "tournament_size": 10
    },
    "MUTATION_PARAM": {
        "type": "uniform",
        "rate": 0.01
    },
    "CROSSOVER_PARAM": {
        "type": "uniform"
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

# chromosome = "01011101000110011000100100011000010001010011000001011001011110110011001101011100100010011110101000110010010101000000110101011100111010011110010101100110011010110100101010111001011100010100100001010001010101100000111111001100001110111000100100011001100100011100100101100010101011001110001100110001100000111111101101101001011110010100110010111010011001110001001101001100101001001011000100010010000010000100000010110010001000110100110001011000011010110010010011100011100101110010010101001011000111000101001111100100000101010000010111000111111011101110110001110001101010010011101010101100011110000110101000101011011100111010001010100101110101011100001110111100101100100110001100001111101010001100011011111001000001110010001101101001010110101110100000010000110110000011000000111100100100111101011011010010010011000010110001111011010001101111011011001111101001101000001001111111110110011001101101110100110100111001110010101011111110000100010011010101001000001001100010101100011111000101010101111000110001010011110001101011101010101110101000100110101000110100100010010000111111101011101100001000000101000100010110011110111010100111111100000100110010110000100110010011100010011000110110100100010011110010100101111110001010101010101010111100101100110101101100001110000010011010000000101001110010110110000000000000110101111010101110101100011100101010101101000100010111100100100110"
#
# network = NetworkFromWeights_2(chromosome, v_max=10.0)
# # Run simulation
# win, environment_surface, agent, font, env = _init_GUI(num_landmarks,
#                                                        num_sensor,
#                                                        sensor_length)
#
# max_time_steps = 1000000
# dust_remains = simulation_network.run_network_simulation(delta_t,
#                                                          max_time_steps,
#                                                          network,
#                                                          agent,
#                                                          win,
#                                                          environment_surface,
#                                                          env,
#                                                          font,
#                                                          True)
#
# print(f"Dust remains: {GeneticAlgorithm(**params).cal_fitness(*dust_remains)}")

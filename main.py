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

delta_t = 2
track = False
num_landmarks = 0
num_sensor = 8
sensor_length = 100

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
    "CHROMOSOME_LENGTH": 106*5,
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


# def main():
#     # Your existing code that initializes and starts the genetic algorithm
#     genAlg = GeneticAlgorithm(**params)
#     genAlg.genetic_algorithm()
#
#
# if __name__ == '__main__':
#     # This line is critical for multiprocessing under Windows
#     from multiprocessing import freeze_support
#
#     freeze_support()
#     main()

########################################################################################################################

# Run the network simulation

# Large
# MaZe
# 110000010001101011100010011111010110000010001111000110101010011100101001101111010000110101011101100101101000111011110000001010100010000100001000010001101011111110001100000000111010101011000110010101001111000101001101010110110100011111110011100100111100110010011100011010011111101110010110100111011010010101010001111101000100010100100001001101001100001000111011000111100110011101011000110010011010010011110111001011101101000101011100011010001001110001001010000001011001011001100111001000101010001010001001100100100000101000111010111101101101110110101110000000001101100000110100011010100001001010001011110100000011001001011110111010101010010011001101100100110001101100001110010001010110111101011110010110111001001111100110001110111111000110101100100001100111101110110110011101100111011111011110010010010100000101100101011000111001101011101110110011010111100011101000100000101111010100100000000000111100000010100110000000100100010101100010100001011011001000101100101111010010010100101110001110101010011100000111100000111000000100011101111111000011101000

# Standard
# 110000010001101011100010011110010110000010001111000110101010011000101001100011010010110101011101100100111100111011010100001010100010000100001000010001111011111110101100000000101010101011000110010101001111000101001001010110110100011111110011000100111100110011011100011010011111101110010100100101011010010011010000111101100100010100100001001101101100001000011011000111100110011111011000110010011010010011110111001011101101000101011000011010000101110000001010000001011001010001100111001000101010001010001001100100110000101000111010111101101101110110101110000000001101100011110100011010100001001010001011110100000011001001011110111010101010000011001101100100110001101100001110010001010110111101011110010110111001001111100110001110111111000110101100100001100101101110110110011101100110011111011010010010010100000111100101011000111001101011101100110011010111100001101000110000101111010100100000000000011110000010100110000000100100010101100010100001011011001000101100101111000010010100101110001110101010011100000111100100111100000100111101011111000011101000

# Medium


# Small
# Standard 11100000001010011110010111011010100010000010110101111011111110111110011011111010101101100011110101010011111110010101001001111000000011101111001000101111010100011111000100010101101000101010111001111011011001001001100000011000011010100100111111100000010010111111000111100011001010010011001100001111100111111101111101011001101000110010101011111001100011001101101101010001001110010101001000000011100010001000000001100000111001011001100010011011000110000110101010101011111111010101001111101001010111111011000100100010100111010110000101

# Maze 00010000101110010101100011010000011001010011001011110101110010011110000001100100010011001000110100011101111101101011111101100000111001000111011101011111111011001100010011101000001010101011101001110000110011010011101010111001001000110101101010010101011000111010100001111011011101001011010001101101100101110010101011010110111011100001011000010011011101000011010101110100110101010010111100100000101011010010111101100101111011100010001111111011000010111010011010010010101001011010010011101001000010010100010010111111001011110011001111

chromosome = "11110000001010011110010111011010100010000000110101111011111110111110011011111010101101100011110111000011111110010101001001111000000011101111001000101111010100011111000100010101101000101010111001110011011001001001100000011000011010100100111111100001010010111111000111100011001010010011001100001111100111111101111101011001101000110010101011111001100011001101101101010001001110010101001000000011100010001001000001100000111001010001100010010011000110000110001010101011111111010101001111101000010111111011010110100010100111110110000101"

network = NetworkFromWeights_2(chromosome, v_max=10.0)
# Run simulation
win, environment_surface, agent, font, env = _init_GUI(num_landmarks,
                                                       num_sensor,
                                                       sensor_length)

max_time_steps = 1000000
dust_remains = simulation_network.run_network_simulation(delta_t,
                                                         max_time_steps,
                                                         network,
                                                         agent,
                                                         win,
                                                         environment_surface,
                                                         env,
                                                         font,
                                                         True)

print(f"Dust remains: {dust_remains}")

from init import simulation_saved, simulation, simulation_network
from src.utils.geneticAlg import GeneticAlgorithm
from src.agent.network import NetworkFromWeights, NetworkFromWeights_2
from init.utils.utils import _init_GUI
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
    "POP_SIZE": 5,
    "CHROMOSOME_LENGTH": 274*5,
    "GEN_MAX": 1,
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
        "type": "two_point"
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

chromosome = "11100110110111101001011100011100100000111101111011011100100101101000110101001101010110010011100010110110011000011000000001001001100110000010111110001000000111000000001010110110101101100100000010100001001001011011000010011100100011011000001001110001010010010100001010001111100101001111111010010101111111010110110111110100100110100101110011001100100110100000001111110111010100011001000100010101100001110111001001101100100101010010011011010000110010001101100000101100111100011111100000110101001001101010011101011110011100101101001110111111111110011110000100100000000010100011000111010110011001110001110100011010101011110010000111101110100000101001111000000011100100001011110111000011100111001011100111110110011110001110111011101011110010011101010010111010100001111011001000000010000000100101111101100010100010010101010010100011111010011011010011111001001101000100001110011110110010000100100010011000000011100111000000111101100001100111000110011100000111100011111001010000001011000010001101110011010011100101110101010110000000011010011111001011111011010111011000100011000011001101011000010111101010111101110001011010000010100010010100100100101100000000101011110000011010010100101011011011001000001000000100000100011101011100001111111011100110110000010010010001010110011111001111101011111100101010010001001111010001111001010010011110100110001010000011110000000111011000010010"

network = NetworkFromWeights_2(chromosome, v_max=10.0)
# Run simulation
win, environment_surface, agent, font, env = _init_GUI(num_landmarks,
                                                       num_sensor,
                                                       sensor_length)

max_time_steps = 500
dust_remains = simulation_network.run_network_simulation(delta_t,
                                                         max_time_steps,
                                                         network,
                                                         agent,
                                                         win,
                                                         environment_surface,
                                                         env,
                                                         font,
                                                         True)

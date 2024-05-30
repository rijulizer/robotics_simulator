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
    "POP_SIZE": 10000,
    "CHROMOSOME_LENGTH": 274*5,
    "GEN_MAX": 10,
    "SELECTION_PARAM": {
        "type": "roulette",
        "num_individuals": 20,
        "tournament_size": 5
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

# chromosome = "10101001100111000000011100000010011100001001101100010011110001100110000111100110010110100111101011110100000011101111111010010111010100001011110111100001001011010000001110011011111110000101001000111001001010100000001010110010001010111010100110111100000001101001011001110100010011011000000100111011101000000100101010111100001001101101011010000100001110100111000000010110101101111101110010000100110100111000010100000111001101010101010110111110010000100101101100010001010100100100011110011101011111101100000000010110010101000010001110010001011101100000011001111000110110011100100001101111000000110000000101101101100011101011110100100111011000011000000100100001010111111100101100001001100111101110001111101001010000000110000010011111011010111111110000000001100100100111110011101110011100010011010100000010011010100110001001000010101011011000100000010010110001001100110000011011111010011011101100110101101110111100111100000111111000110001000001000001001110110001101111111101110011101011010111001000010001111111011111110100101010101101111110100001111111010101111100000010011101111100011100100111001101110010011111010111111100101100110101001010010000111110000101000011100010110010100111111011000110010110110100111001110101001000111100101110001011010101010000000000011001100010110100110011110010101010011000111100010111111010000101011000001000001110010110010000011101110100010100"
#
# network = NetworkFromWeights_2(chromosome, v_max=10.0)
# # Run simulation
# win, environment_surface, agent, font, env = _init_GUI(num_landmarks,
#                                                        num_sensor,
#                                                        sensor_length)
#
# max_time_steps = 500
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

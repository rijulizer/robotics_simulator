from init import simulation_saved, simulation, simulation_network
from src.utils.geneticAlg import GeneticAlgorithm
from src.agent.network import NetworkFromWeights
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
    "POP_SIZE": 100,
    "CHROMOSOME_LENGTH": 240,
    "GEN_MAX": 50,
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

# chromosome = "111111001010001011001000010111101111011001010110000001110001101100101111111101100000100011111011100011010110110111111101111010110000111011101111011101011001111001001101101000010010001000011111101011001011000011100100111101110100111010101010"
#
# network = NetworkFromWeights(chromosome, v_max=10.0)
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

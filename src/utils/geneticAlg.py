import random
import numpy as np
import pygame
import sys
import multiprocessing
from multiprocessing import Pool

from init.utils.utils import _init_GUI
from src.agent.network import NetworkFromWeights, NetworkFromWeights_2
from init.simulation_network import run_network_simulation
import tqdm.auto as tqdm
import heapq

MAX_VELOCITY = 5.0


class GeneticAlgorithm:

    def __init__(self,
                 POP_SIZE,
                 CHROMOSOME_LENGTH,
                 GEN_MAX,
                 SELECTION_PARAM,
                 MUTATION_PARAM,
                 CROSSOVER_PARAM):

        self.POP_SIZE = POP_SIZE
        self.CHROMOSOME_LENGTH = CHROMOSOME_LENGTH
        self.GEN_MAX = GEN_MAX
        self.SELECTION_PARAM = SELECTION_PARAM
        self.MUTATION_PARAM = MUTATION_PARAM
        self.CROSSOVER_PARAM = CROSSOVER_PARAM

    def __str__(self):
        settings = f"Population Size: {self.POP_SIZE}\n"
        settings += f"Chromosome Length: {self.CHROMOSOME_LENGTH}\n"
        settings += f"Maximum Generations: {self.GEN_MAX}\n"
        settings += f"Selection Parameters: {self.SELECTION_PARAM}\n"
        settings += f"Mutation Parameters: {self.MUTATION_PARAM}\n"
        settings += f"Crossover Parameters: {self.CROSSOVER_PARAM}\n"
        return settings

    # Helper function to create a random gene
    def random_gene(self):
        return random.choice(['0', '1'])

    # Create one random chromosome
    def random_chromosome(self):
        return {"Gen": ''.join(self.random_gene() for _ in range(self.CHROMOSOME_LENGTH)), "fitness": -1}

    def fitness(self, chromosome):
        if chromosome["fitness"] == -1:
            raise ValueError("Fitness not calculated")
        return chromosome["fitness"]

    # Tournament Selection
    def _tournament_selection(self, population):
        selected = []
        for _ in range(self.SELECTION_PARAM["num_individuals"]):  # Select k individuals
            contenders = random.sample(population, self.SELECTION_PARAM["tournament_size"])  # Tournament size
            selected.append(max(contenders, key=self.fitness))
        return selected

    # Roulette Wheel Selection
    def _roulette_selection(self, population):
        total_fitness = sum(self.fitness(chromo) for chromo in population)
        selection_probs = [self.fitness(chromo) / total_fitness for chromo in population]

        return random.choices(population, weights=selection_probs, k=self.SELECTION_PARAM["num_individuals"])

    def selection(self, population):
        if self.SELECTION_PARAM['type'] == 'tournament':
            return self._tournament_selection(population)
        elif self.SELECTION_PARAM['type'] == 'roulette':
            return self._roulette_selection(population)
        else:
            raise ValueError("Invalid selection type")

    # One point crossover
    def _one_point_crossover(self, parent1, parent2):
        cross_pt = random.randint(1, self.CHROMOSOME_LENGTH - 1)
        child1 = parent1[:cross_pt] + parent2[cross_pt:]
        child2 = parent2[:cross_pt] + parent1[cross_pt:]
        return child1, child2

    # Two point crossover
    def _two_point_crossover(self, parent1, parent2):
        pt1, pt2 = sorted(random.sample(range(1, self.CHROMOSOME_LENGTH), 2))
        child1 = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
        child2 = parent2[:pt1] + parent1[pt1:pt2] + parent2[pt2:]
        return child1, child2

    # Uniform Crossover
    def _uniform_crossover(self, parent1, parent2):
        child1, child2 = '', ''
        for i in range(self.CHROMOSOME_LENGTH):
            if random.random() < 0.5:
                child1 += parent1[i]
                child2 += parent2[i]
            else:
                child1 += parent2[i]
                child2 += parent1[i]
        return child1, child2

    def crossover(self, parent1, parent2):
        self.CROSSOVER_PARAM['type'] = random.choice(['one_point', 'two_point', 'uniform'])

        if self.CROSSOVER_PARAM['type'] == 'one_point':
            return self._one_point_crossover(parent1, parent2)
        elif self.CROSSOVER_PARAM['type'] == 'two_point':
            return self._two_point_crossover(parent1, parent2)
        elif self.CROSSOVER_PARAM['type'] == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        else:
            raise ValueError("Invalid crossover type")

    # Mutation - Uniform Mutation
    def _uniform_mutate(self, chromosome):
        chromosome = list(chromosome)
        for i in range(len(chromosome)):
            if random.random() < self.MUTATION_PARAM['rate']:
                chromosome[i] = self.random_gene()
        return ''.join(chromosome)

    # Mutation - Bit Flip Mutation
    def _bit_flip_mutate(self, chromosome):
        chromosome = list(chromosome)
        for i in range(len(chromosome)):
            if random.random() < self.MUTATION_PARAM['rate']:
                chromosome[i] = '1' if chromosome[i] == '0' else '0'
        return ''.join(chromosome)

    def mutate(self, chromosome):
        if self.MUTATION_PARAM['type'] == 'uniform':
            mutated = self._uniform_mutate(chromosome)
        elif self.MUTATION_PARAM['type'] == 'bit_flip':
            mutated = self._bit_flip_mutate(chromosome)
        else:
            raise ValueError("Invalid mutation type")

        return {"Gen": mutated, "fitness": -1}

    def simulate_chromosome(self, chromo):
        num_landmarks = 0
        num_sensor = 8
        sensor_length = 100
        delta_t = 2
        max_time_steps = 100000000000

        network = NetworkFromWeights_2(chromo["Gen"], MAX_VELOCITY * 2)
        win, environment_surface, agent, font, env = _init_GUI(num_landmarks, num_sensor, sensor_length,
                                                               pygame_flags=pygame.HIDDEN)
        results = run_network_simulation(delta_t, max_time_steps, network, agent, win, environment_surface, env, font)

        return chromo["Gen"], results

    def genetic_algorithm(self):
        # Open file to write the results
        print("Number of CPUs: ", multiprocessing.cpu_count())

        generation = 0

        progress_bar = tqdm.tqdm(total=self.GEN_MAX * self.POP_SIZE, desc="Running Genetic Algorithm",
                                 file=sys.stdout)

        population = [self.random_chromosome() for _ in range(self.POP_SIZE)]

        # gen_1 = "11100000001010011110010111011010100010000010110101111011111110111110011011111010101101100011110101010011111110010101001001111000000011101111001000101111010100011111000100010101101000101010111001111011011001001001100000011000011010100100111111100000010010111111000111100011001010010011001100001111100111111101111101011001101000110010101011111001100011001101101101010001001110010101001000000011100010001000000001100000111001011001100010011011000110000110101010101011111111010101001111101001010111111011000100100010100111010110000101"
        # gen_2 = "11110000001010011110010111011010100010000000110101111011111110111110011011111010101101100011110111000011111110010101001001111000000011101111001000101111010100011111000100010101101000101010111001110011011001001001100000011000011010100100111111100001010010111111000111100011001010010011001100001111100111111101111101011001101000110010101011111001100011001101101101010001001110010101001000000011100010001001000001100000111001010001100010010011000110000110001010101011111111010101001111101000010111111011010110100010100111110110000101"
        #
        # population.append({"Gen": gen_1, "fitness": -1})
        # population.append({"Gen": gen_2, "fitness": -1})

        while generation < self.GEN_MAX:
            with Pool(multiprocessing.cpu_count()) as pool:
                # Iterate by batches of 100
                for i in range(0, len(population), 100):
                    results = pool.map(self.simulate_chromosome, population[i:i + 100])
                    for j, (chromo, fitness) in enumerate(results):
                        population[i + j]["Gen"] = chromo
                        population[i + j]["fitness"] = fitness
                    progress_bar.update(len(results))

            best_samples = heapq.nlargest(5, population, key=lambda x: self.fitness(x))
            with open('src/data/genEvo/genetic_algorithm_results_1.txt', 'a') as file:
                for sample in best_samples:
                    file.write(f"Gen: {generation}, Fitness: {sample['fitness']}, Best Sample: {sample['Gen']}\n")
                file.close()

            # population = self.selection(population)
            new_generation = []

            new_generation.extend(best_samples)

            while len(new_generation) < self.POP_SIZE:
                parent1, parent2 = random.sample(best_samples, 2)
                child1, child2 = self.crossover(parent1["Gen"], parent2["Gen"])
                new_generation.append(self.mutate(child1))
                new_generation.append(self.mutate(child2))

            population = new_generation

            generation += 1

        progress_bar.close()

    def run_print(self, dust_collect, unique_positions, energy_used, rotation_measure, num_avg_collision,
                  final_fitness):
        print(f"\nDust Collected: {dust_collect}")
        print(f"Unique Positions: {unique_positions}")
        print(f"Energy Used: {energy_used}")
        print(f"Rotation Measure: {rotation_measure}")
        print(f"Average Collision: {num_avg_collision}")
        print(f"Final Fitness: {final_fitness}")

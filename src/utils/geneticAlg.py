import random
from pprint import pprint

from init.utils.utils import _init_GUI
from src.agent.network import NetworkFromWeights
from init.simulation_network import run_network_simulation

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

    # Fitness function
    def fitness(self, chromosome):
        if chromosome["fitness"] == -1:
            raise ValueError("Fitness not calculated")
        dust_collect = chromosome["fit_dust_collect"]
        unique_positions = chromosome["fit_uniqe_pos"] 
        energy_used  = chromosome["fit_energy_used"]
        chromosome["fitness"] = dust_collect + unique_positions + energy_used
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

        selected = random.choices(
            population, weights=selection_probs, k=self.SELECTION_PARAM["num_individuals"]  # Select k individuals
        )
        return selected

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
            return self._uniform_mutate(chromosome)
        elif self.MUTATION_PARAM['type'] == 'bit_flip':
            return self._bit_flip_mutate(chromosome)
        else:
            raise ValueError("Invalid mutation type")

    def genetic_algorithm(self):
        population = [self.random_chromosome() for _ in range(self.POP_SIZE)]

        # Open file to write the results
        with open('src/data/genEvo/genetic_algorithm_results.txt', 'w') as file:
            generation = 0

            # Initialize simulation parameters
            num_landmarks = 0
            num_sensor = 12
            sensor_length = 100
            delta_t = 1
            max_time_steps = 1000

            while generation < self.GEN_MAX:
                # Evaluate our population (Calculate fitness)
                for chromo in population:
                    network = NetworkFromWeights(chromo["Gen"], MAX_VELOCITY * 2)
                    # Run simulation
                    win, environment_surface, agent, font, env = _init_GUI(num_landmarks,
                                                                           num_sensor,
                                                                           sensor_length)

                    (
                        dust_collect, 
                        unique_positions, 
                        energy_used,
                    )   = run_network_simulation(delta_t,
                                                          max_time_steps,
                                                          network,
                                                          agent,
                                                          win,
                                                          environment_surface,
                                                          env,
                                                          font)

                    chromo["fit_dust_collect"] = dust_collect
                    chromo["fit_uniqe_pos"] = unique_positions
                    chromo["fit_energy_used"] = energy_used


                    pprint(chromo)


                best_sample = max(population, key=self.fitness)
                file.write(f"Gen: {generation}, Best Sample: {best_sample}, Fitness: {self.fitness(best_sample)}\n")
                if self.fitness(best_sample) == 1:
                    break

                population = self.selection(population)
                new_generation = []

                while len(new_generation) < self.POP_SIZE:
                    parent1, parent2 = random.sample(population, 2)
                    child1, child2 = self.crossover(parent1["Gen"], parent2["Gen"])
                    new_generation.append(self.mutate(child1))
                    new_generation.append(self.mutate(child2))

                population = new_generation
                # Reset fitness in population
                for chromo in population:
                    chromo["fitness"] = -1
                generation += 1

        return best_sample

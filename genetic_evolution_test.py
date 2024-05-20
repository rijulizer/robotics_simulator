
from main import run_network_simulation
from src.neuralnetwork import NeuralNetwork
import numpy as np
import pickle as pkl

class GeneticEvolution:
    def __init__(self):
        self.learning_rate = [0.0,0.3,0.7,1.0]

    def bin2int(self,num):
        return num[1]*2+num[0]

    def generate_random_sequence(self,sequence):
        #sequence = np.random.choice([0, 1], size=(45,6))
        str_seq = "".join(str(s) for s in sequence)
        return str_seq,(sequence[0],sequence[1],self.bin2int(sequence[2:4]),self.bin2int(sequence[4:6]))
    
    def generate_single_sequence(self):
        learning_rate = np.zeros((45))
        learning_rule = np.zeros((45),dtype=int)
        driving_rule = np.zeros((45),dtype=int)
        excitatory_rule = np.zeros((45),dtype=int)
        final_seq = ""
        #sequence = np.random.choice([0, 1], size=(45,6))
        sequence = np.random.randint(2,size = (45,6))
        for i in range(45):
            seq,seq_data = self.generate_random_sequence(sequence[i])
            final_seq += seq
            learning_rate[i] = self.learning_rate[seq_data[3]]
            learning_rule[i] = seq_data[2]
            driving_rule[i] = seq_data[0]
            excitatory_rule[i] = seq_data[1] if seq_data[1] == 1 else -1
        return sequence,final_seq,driving_rule,excitatory_rule,learning_rate,learning_rule

    def run_test_evolution1(self):
        number_sensors = 12
        population_size = 100
        data = []
        datapkl = []
        for p in range(population_size):
            nseq,seq,driving_rule,excitatory_rule,learning_rate,learning_rule = self.generate_single_sequence()
            nn = NeuralNetwork(number_sensors,driving_rule,excitatory_rule,learning_rate,learning_rule)
            res = run_network_simulation(delta_t=1,
                           graphGUI=None,
                           track=False,
                           num_landmarks=0,
                           max_time_steps = 160,
                           network = nn)
            data.append((seq,res))
            datapkl.append((nseq,res))
            print (p," population running ", end='\r')
        print ("")
        with open(f'./src/experiments_data/genetic_evolution_results.txt', 'w') as f:
            for s,r in data:
                f.write(f"{s},{r}\n")
        with open(f'./src/experiments_data/genetic_evolution_resultsb.pkl', 'wb') as f:
            pkl.dump(datapkl, f)       

if __name__ == "__main__":

    ge = GeneticEvolution()
    #print(ge.generate_random_sequence())
    ge.run_test_evolution1()


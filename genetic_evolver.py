from main import run_network_simulation
from src.neuralnetwork import NeuralNetwork
import numpy as np
import pickle as pkl

TOTAL_NUMBER_NEURONS = (12+3)*3

# driving or modulatory (1 bit)
# excitatory or inhibitory (1 bit)
# learning rule (2 bits) 
# learning rate (2 bits) --> [0.0,0.3,0.7,1.0]
GENETIC_PARAMS_PER_NEURON = 6
NUMBER_OF_SENSORS = 12

class GeneticEvolver:
    def __init__(self):
        self.learning_rate = [0.0,0.3,0.7,1.0]
        self.record = None

    def bin2dec(self,num):
        return num[1]*2+num[0]

    def read_test_file(self,file):
        with open(file, "rb") as f:
            self.record = pkl.load(f)

    def decode_sub_sequence(self,sequence):
        str_seq = "".join(str(s) for s in sequence)
        return str_seq,(sequence[0],sequence[1],self.bin2dec(sequence[2:4]),self.bin2dec(sequence[4:6]))

    def generate_single_sequence(self):
        learning_rate = np.zeros((TOTAL_NUMBER_NEURONS))
        learning_rule = np.zeros((TOTAL_NUMBER_NEURONS),dtype=int)
        driving_rule = np.zeros((TOTAL_NUMBER_NEURONS),dtype=int)
        excitatory_rule = np.zeros((TOTAL_NUMBER_NEURONS),dtype=int)
        final_seq = ""
        sequence = np.random.randint(2,size = (TOTAL_NUMBER_NEURONS,GENETIC_PARAMS_PER_NEURON))
        for i in range(45):
            seq,seq_data = self.decode_sub_sequence(sequence[i])
            final_seq += seq
            learning_rate[i] = self.learning_rate[seq_data[3]]
            learning_rule[i] = seq_data[2]
            driving_rule[i] = seq_data[0]
            excitatory_rule[i] = seq_data[1] if seq_data[1] == 1 else -1
        return sequence,final_seq,driving_rule,excitatory_rule,learning_rate,learning_rule

    def decode_single_sequence(self,sequence):
        learning_rate = np.zeros((TOTAL_NUMBER_NEURONS))
        learning_rule = np.zeros((TOTAL_NUMBER_NEURONS),dtype=int)
        driving_rule = np.zeros((TOTAL_NUMBER_NEURONS),dtype=int)
        excitatory_rule = np.zeros((TOTAL_NUMBER_NEURONS),dtype=int)
        final_seq = ""
        for i in range(TOTAL_NUMBER_NEURONS):
            seq,seq_data = self.decode_sub_sequence(sequence[i])
            final_seq += seq
            learning_rate[i] = self.learning_rate[seq_data[3]]
            learning_rule[i] = seq_data[2]
            driving_rule[i] = seq_data[0]
            excitatory_rule[i] = seq_data[1] if seq_data[1] == 1 else -1
        return final_seq,driving_rule,excitatory_rule,learning_rate,learning_rule

    def reproduce_members(self,population_size):
        probabilities = np.array([c for _,c in self.record])
        probabilities = probabilities/np.sum(probabilities)

        newgenes_index = np.random.choice(range(len(self.record)),size = (population_size),p=probabilities)
        pairs = np.random.choice(range(len(self.record)),size = (population_size),replace=False)
        pairs_cat1 = [pairs[i] for i in range(0,len(pairs),2)]
        pairs_cat2 = [pairs[i] for i in range(1,len(pairs),2)]
        crossover_site = np.random.choice(range(self.record[0][0].size),size=(int(population_size/2)))
        mutationprob = 0.2
        mutation_no = int(mutationprob*self.record[0][0].size*population_size)
        mutation_indices = np.sort((np.random.choice(range(self.record[0][0].size*population_size),size=(mutation_no))),axis=None)
        
        selected_genes = np.array([self.record[i][0].reshape(-1) for i in newgenes_index])
        for i,(p1,p2) in enumerate(zip(pairs_cat1,pairs_cat2)):
            #print(i,p1,p2,crossover_site[i],selected_genes[p1].shape)
            newgene1 = np.concatenate((selected_genes[p1][:crossover_site[i]-1],selected_genes[p2][crossover_site[i]-1:]))
            newgene2 = np.concatenate((selected_genes[p2][:crossover_site[i]-1],selected_genes[p1][crossover_site[i]-1:]))
            selected_genes[p1] = newgene1
            selected_genes[p2] = newgene2
        for mi in mutation_indices:
            index1 = int(mi / 270)
            index2 = int(mi % 270)
            selected_genes[index1][index2] = selected_genes[index1][index2] ^ 1
        
        selected_genes = selected_genes.reshape((population_size,TOTAL_NUMBER_NEURONS,GENETIC_PARAMS_PER_NEURON))
        return selected_genes

    def run_test(self,fitness_value):
        interested_s= None
        for s,c in self.record:
            if c==fitness_value:
                interested_s = s
                break
        
        number_sensors = NUMBER_OF_SENSORS
        _,driving_rule,excitatory_rule,learning_rate,learning_rule = self.decode_single_sequence(interested_s)
        nn = NeuralNetwork(number_sensors,driving_rule,excitatory_rule,learning_rate,learning_rule)
        run_network_simulation(delta_t=1,
                       graphGUI=None,
                       track=False,
                       num_landmarks=0,
                       max_time_steps = 5000,
                       network = nn,
                       pygameflags=None)
        
    def run_test_evolution_starter(self, filename = './src/experiments_data/genetic_evolution_results'):
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
        
        txtfile = filename + ".txt"
        binfile = filename + "b.pkl"
        with open(txtfile, 'w') as f:
            for s,r in data:
                f.write(f"{s},{r}\n")
        with open(binfile, 'wb') as f:
            pkl.dump(datapkl, f)

    def run_test_evolutionn(self,filename):
        population_size = 100
        newgenes = self.reproduce_members(population_size)
        number_sensors = 12
        data = []
        datapkl = []
        for p in range(population_size):
            seq,driving_rule,excitatory_rule,learning_rate,learning_rule = self.decode_single_sequence(newgenes[p])
            nn = NeuralNetwork(number_sensors,driving_rule,excitatory_rule,learning_rate,learning_rule)
            res = run_network_simulation(delta_t=1,
                           graphGUI=None,
                           track=False,
                           num_landmarks=0,
                           max_time_steps = 160,
                           network = nn)
            data.append((seq,res))
            datapkl.append((newgenes[p],res))
            print (p," population running ", end='\r')
        print ("")
        txtfile = filename + ".txt"
        binfile = filename + "b.pkl"
        with open(txtfile, 'w') as f:
            for s,r in data:
                f.write(f"{s},{r}\n")
        with open(binfile, 'wb') as f:
            pkl.dump(datapkl, f)     
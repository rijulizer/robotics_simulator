from genetic_evolver import GeneticEvolver

# if __name__ == "__main__":
#     ge = GeneticEvolver()
#     ge.read_previous_test_file('./src/experiments_data/genetic_evolution_results10b.pkl')
#     ge.run_test_evolutionn('./src/experiments_data/genetic_evolution_results11')

if __name__ == "__main__":
    for i in range(1,50,1):
        ge = GeneticEvolver()
        filename = './src/experiments_data/genetic_evolution_results'+str(i)+'b.pkl'
        binname = './src/experiments_data/genetic_evolution_results'+str(i+1)
        ge.read_test_file(filename)
        ge.run_test_evolutionn(binname)
from genetic_evolver import GeneticEvolver
        
if __name__ == "__main__":
        ge = GeneticEvolver()
        running_case = ['./src/experiments_data/genetic_evolution_results39b.pkl','9'] #ok
        #running_case = ['./src/experiments_data/genetic_evolution_results22b.pkl','9']
        #running_case = ['./src/experiments_data/genetic_evolution_results11b.pkl','9'] ok
        ge.read_test_file(running_case[0])
        ge.run_test(int(running_case[1]))

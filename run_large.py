import os
import argparse


from evrp.evrp_instance import EvrpInstance
from evrp.ga import *


BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname('.')))

DATA_DIR = os.path.join(BASE_DIR, 'evrp-benchmark-set')
RESULT_DIR = os.path.join(BASE_DIR, 'results')


def run_large_instance(instance, seed, pop_size, n_gen, cx_prob, mut_prob, indpb, result_dir, is_export_csv=True):
    random.seed(seed)
    
    CANDIDATES = PriorityQueue()
    PLAIN_CANDIDATES_SET = []

    num_vehicles =  instance.num_of_vehicles
    num_customers = instance.dimension - 1
    capacity = instance.capacity
    demands = instance.demands
    station_list = instance.station_list
    
    csv_data = []
    
    start_time = time.process_time()
    # population initialization
    pop = []
    # init_pop_randomly = [generate_individual_randomly(num_vehicles, num_customers) for _ in range(int(pop_size * 0.8))]
    init_pop_randomly = [generate_individual_randomly_wisely(num_vehicles, num_customers, capacity, demands) for _ in range(int(pop_size * 0.9))]
    init_pop_randomly_wisely = [sublist for sublist in init_pop_randomly if sublist] 
    init_pop_evenly = [generate_individual_evenly(num_vehicles, num_customers) for _ in range(int(pop_size * 0.2))]
    pop.extend(init_pop_randomly_wisely)
    pop.extend(init_pop_evenly)
    
    best_solution = []
    best_cost = 0.0
    
    print('----------------Start of evolution----------------')
    for gen in range(n_gen):
        print(f'---- Generation {gen} ----')
        
        offspring_after_crossover = []
        for ind1, ind2 in zip(pop[::2], pop[1::2]):
            if random.random() < cx_prob:
                child1, child2 = cx_partially_matched(ind1, ind2)
                offspring_after_crossover.append(child1)
                offspring_after_crossover.append(child2)        

        offspring_after_muatation = []
        for ind in offspring_after_crossover:
            if random.random() < mut_prob:
                mutant = mut_shuffle_indexes(ind, indpb)
                offspring_after_muatation.append(mutant)

        stats_num_candidates_added = 0
        pop_pool = pop + offspring_after_crossover + offspring_after_muatation
        for individual in pop_pool:
            if is_capacity_feasible(individual, capacity, demands):
                if individual not in PLAIN_CANDIDATES_SET:
                    stats_num_candidates_added += 1
                    PLAIN_CANDIDATES_SET.append(individual)
                    optimized_individual, cost = local_search(individual, instance)
                    CANDIDATES.push(optimized_individual, cost)
        print(f'  Evaluated {stats_num_candidates_added} individuals')
        candidates_size = CANDIDATES.size()
        print(f'  Candidates Pool Size: {candidates_size}')

        
        # Elites Population
        elites = CANDIDATES.peek(1000)

        # Statistical Data
        size = len(elites)
        fits = []
        mean = 0
        std  = 0.0
        min_fit = None
        max_fit = None

        if size == 0:
            print('  No candidates')
        else:
            fits = [fit for fit, idx, ind in elites]
            mean = sum(fits) / size
            min_fit = min(fits)
            max_fit = max(fits)

        if size > 1:
            std = (sum((x - mean) ** 2 for x in fits) / (size - 1)) ** 0.5
        else:
            std = 0.0
        print(f'  Elites Num {size}')
        print(f'  Min {min_fit}') # the best result of each generation
        print(f'  Max {max_fit}')
        print(f'  Avg {mean}')   # Reflect the direction of population evolution 
        print(f'  Std {std}')
        
        min_individual = [] 
        min_fitness = None
        if size != 0:
            min_fitness, idx, min_individual = CANDIDATES.peek(1)[0]
        print(f'  Best fitness: {min_fit}')
        best_solution = min_individual
        best_cost = min_fitness
        # Write data to holders for exporting results to CSV file
        if is_export_csv:
            csv_row = {
                'generation': gen,
                'evaluated_individuals': stats_num_candidates_added,
                'candidates_size': candidates_size,
                'min_fitness': min_fit,
                'max_fitness': max_fit,
                'avg_fitness': mean,
                'std_fitness': std,
                'duration': elapsed_time(start_time),
            }
            csv_data.append(csv_row)
    
        # select
        pop = []  
        elites = [ind for fit, idx, ind in elites[:500]]
        elites.extend(CANDIDATES.random_elements(1000))
        elites_without_stations = []
        for ind in elites:
            individual_without_station = []
            for route in ind:
                route_without_stations = [node for node in route if str(node) not in station_list]
                individual_without_station.append(route_without_stations)
            elites_without_stations.append(individual_without_station)       
        pop.extend(elites_without_stations)
        
        num_left = pop_size - len(elites_without_stations)
        random.shuffle(pop_pool)
        individuals_from_pop_pool = pop_pool[:int(num_left * 0.7)]
        
        individuals_from_immigration = []
        individuals_evenly = [generate_individual_evenly(num_vehicles, num_customers) for _ in range(int(num_left * 0.1))]
        # individuals_randomly = [generate_individual_randomly(num_vehicles, num_customers) for _ in range(int(num_left * 0.2))]
        individuals_randomly = [generate_individual_randomly_wisely(num_vehicles, num_customers, capacity, demands) for _ in range(int(num_left * 0.2))]
        individuals_randomly_wisely = [sublist for sublist in individuals_randomly if sublist]  
        individuals_from_immigration.extend(individuals_evenly)
        individuals_from_immigration.extend(individuals_randomly_wisely)
        
        pop.extend(individuals_from_pop_pool)
        pop.extend(individuals_from_immigration)

    print('------------End of (successful) evolution------------', end='\n\n') 
    duration = elapsed_time(start_time)
    print(f'The Total Evolutionary Duration: {duration} for {n_gen} generations')
    print(f'The average evolutionary time for each generation is: {duration/n_gen}')

    csv_file = ''
    if is_export_csv:
        csv_file_name = f'{instance.name}_seed{seed}__pS{pop_size}_nG{n_gen}.csv'
        csv_file = os.path.join(result_dir, csv_file_name) 
        print(f'Write to file: {csv_file}')
        make_dirs_for_file(path=csv_file)
        if not exist(path=csv_file, overwrite=True):
            with io.open(csv_file, 'wt', newline='') as file_object:
                fieldnames = [
                    'generation',
                    'evaluated_individuals',
                    'candidates_size',
                    'min_fitness',
                    'max_fitness',
                    'avg_fitness',
                    'std_fitness',
                    'duration'
                ]
                writer = DictWriter(file_object, fieldnames=fieldnames, dialect='excel')
                writer.writeheader()
                for csv_row in csv_data:
                    writer.writerow(csv_row)
                    
    return (best_solution, best_cost, csv_file)
        
def main():
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--instance_name','-n',type=str, default = "X-n214-k11.evrp",required=True,help="a instance name")
    parser.add_argument('--seed','-s',type=int, default=1,help='random seed')
    parser.add_argument('--pop_size','-ps',type=int, default=10000,help='population size')
    parser.add_argument('--n_gen','-g',type=int, default=150,help='the training generation number')
    parser.add_argument('--rate_elite','-r',type=float, default=0.2,help='The rate of elites for the population when selecting')
    parser.add_argument('--cx_prob','-c',type=float, default=0.8,help='the probability of crossover')
    parser.add_argument('--mut_prob','-m',type=float, default=0.5,help='the probability of mutation')
    parser.add_argument('--indpb','-ind',type=float, default=0.2,help='the mutation probability for each element in the sequence')
    parser.add_argument('--immig_prop','-i',type=float, default=0.2,help='the proportion of immigrants in the new population')

    args = parser.parse_args()

    instance_name = args.instance_name 
    seed = args.seed
    pop_size = args.pop_size
    n_gen = args.n_gen
    cx_prob = args.cx_prob
    mut_prob = args.mut_prob
    indpb = args.indpb


    file_dir = os.path.join(DATA_DIR, instance_name)
    instance = EvrpInstance(file_dir)

    best_solution, best_cost, training_file = run_large_instance(instance, seed, pop_size, n_gen, cx_prob, mut_prob, indpb, RESULT_DIR)

    df = create_dataframe(instance)


    print(f'Excepted optimum: {instance.optimum}')
    print(f'The optimal result cost from my GA: {best_cost}')
    print(best_solution)

    visualize(training_file)
    visualize_routes(best_solution, df, 'Best Solution for ' + instance.name)


def visualize(training_file):
    df_run = pd.read_csv(training_file)
    title = os.path.basename(training_file).split('_')[0] + ' GA '
    plot_training_graph(df_run, title)


if __name__ == '__main__':
    main()


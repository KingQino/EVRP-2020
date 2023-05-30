# -*- coding: utf-8 -*-
from .ga import *

def simple_local_search(individual, instance):
    '''Incooperate several local search opereators
    
    Args:
        individual: A list of routes. Each route is a list of nodes.
        instance: the benchmark instance got from the original data
    
    Returns:
        a tuple contains a improved individual and the corresponding cost
    '''
    battery_capacity = instance.battery_capacity
    energy_consumption = instance.energy_consumption
    distance_matrix = instance.distance_matrix
    station_list = instance.station_list

    # 2-opt 
    optimized_individual = []
    for route in individual:
        if len(route) <= 1:
            optimized_individual.append(route)
            continue
        optimized_route = two_opt(route, distance_matrix)
        optimized_individual.append(optimized_route)

    # simple repair (ZGA)
    original_individual = optimized_individual
    repaired_individual = []
    for route in original_individual:
        repaired_route = simple_repair(route, battery_capacity, energy_consumption, distance_matrix, station_list)
        repaired_individual.append(repaired_route)


    # station_realloc_1
    improved_individual = station_realloc_1(repaired_individual, battery_capacity, energy_consumption, distance_matrix, station_list)
    cost = fitness_evaluation(improved_individual, distance_matrix)
    
    return (improved_individual, cost)



def pop_init(size, rand_rate, instance_name, num_vehicles, num_customers, capacity, demands):
    num_randomly = int(size * rand_rate)
    num_evenly = int(size * (1- rand_rate))

    pop = []
    if instance_name.startswith('E'):
        init_pop_randomly = [generate_individual_randomly(num_vehicles, num_customers) for _ in range(num_randomly)]
        pop.extend(init_pop_randomly)
    elif instance_name.startswith('X'):
        init_pop_randomly = [generate_individual_randomly_wisely(num_vehicles, num_customers, capacity, demands) for _ in range(num_randomly)]
        init_pop_randomly_wisely = [sublist for sublist in init_pop_randomly if sublist] 
        pop.extend(init_pop_randomly_wisely)
    else:
        raise NameError("The input argument '-n', namely, '--instance_name' is incorrect!")

    init_pop_evenly = [generate_individual_evenly(num_vehicles, num_customers) for _ in range(num_evenly)]

    pop.extend(init_pop_evenly)

    return pop


def run_MA(instance, seed, pop_size, n_gen, cx_prob, mut_prob, indpb, cand_ratio, rand_rate, ng_pop_rate, ng_immig_rate, result_dir, is_export_csv=True):
    random.seed(seed)
    
    CANDIDATES = PriorityQueue()
    PLAIN_CANDIDATES_SET = set()
    CANDIATES_POOL_SIZE = int(cand_ratio * pop_size)

    num_vehicles =  instance.num_of_vehicles
    num_customers = instance.dimension - 1
    capacity = instance.capacity
    demands = instance.demands
    station_list = instance.station_list
    
    csv_data = []
    
    start_time = time.process_time()
    # population initialization
    pop = []
    ancester_pop = pop_init(pop_size, rand_rate, instance.name, num_vehicles, num_customers, capacity, demands)
    pop.extend(ancester_pop)
    

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
                string_individual = '|'.join('-'.join(str(element) for element in sublist) for sublist in individual)
                if string_individual not in PLAIN_CANDIDATES_SET:
                    stats_num_candidates_added += 1
                    PLAIN_CANDIDATES_SET.add(string_individual)
                    optimized_individual, cost = simple_local_search(individual, instance)
                    CANDIDATES.push(optimized_individual, cost)
        print(f'  Evaluated {stats_num_candidates_added} individuals')
        candidates_size = CANDIDATES.size()
        if candidates_size > CANDIATES_POOL_SIZE:
            CANDIDATES.remove_elements(candidates_size - CANDIATES_POOL_SIZE)
            candidates_size = CANDIDATES.size()
            print('  Clear extra candidates from Candiate Pool')
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
            fits = [fit for fit, ind in elites]
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
            min_fitness, min_individual = CANDIDATES.peek(1)[0]
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
        elites = [ind for fit, ind in elites[:500]]
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
        individuals_from_pop_pool = pop_pool[:int(num_left * ng_pop_rate)]
        pop.extend(individuals_from_pop_pool)
        
        immigrants_num = int(num_left * ng_immig_rate)
        immigrants = pop_init(immigrants_num, rand_rate, instance.name, num_vehicles, num_customers, capacity, demands)
        pop.extend(immigrants)


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



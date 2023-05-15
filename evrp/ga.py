# -*- coding: utf-8 -*-
import os
import io
import random
from csv import DictWriter

from .utils import *  



def generate_individual_evenly(num_vehicles, num_customers):
    """Generate an individual by distributing customers evenly across vehicles.
    
    Args:
        num_vehicles: The number of vehicles.
        num_customers: The number of customers.
        
    Returns:
        A list of lists, where each sublist represents a vehicle and its assigned customers.
    """

    # Create a list of customer numbers and shuffle it
    customers = list(range(1, num_customers + 1))

    # Initialize the individual
    individual = []

    random.shuffle(customers)
    individual = [customers[i::num_vehicles] for i in range(num_vehicles)]
    
    return individual


def generate_individual_randomly(num_vehicles, num_customers):
    """Generate an individual by randomly assigning customers to vehicles.
    
    Args:
        num_vehicles: The number of vehicles.
        num_customers: The number of customers.
        
    Returns:
        A list of lists, where each sublist represents a vehicle and its assigned customers.
    """
    
    # Initialize an empty individual with a list for each vehicle
    individual = [[] for _ in range(num_vehicles)]
    
    # Create a list of customers and shuffle it
    customers = list(range(1, num_customers + 1))
    random.shuffle(customers)
    
    for customer in customers:
        # Assign the customer to a random vehicle
        vehicle = random.randint(0, num_vehicles - 1)
        individual[vehicle].append(customer)
    
    return individual



# ************************************************************************
# ***************************** Operators ********************************

# crossover
def cx_partially_matched(ind1, ind2):
    """Partially Matched Crossover (PMX)
    
    Args:
        ind1: The first individual participating in the crossover.
        ind2: The second individual participating in the crossover.
        
    Returns:
        A tuple of two individuals.
    """
    
    route_lengths_1 = [len(route) for route in ind1]
    route_lengths_2 = [len(route) for route in ind2]
    flat_ind1 = [node - 1 for route in ind1 for node in route]
    flat_ind2 = [node - 1 for route in ind2 for node in route]
    
    size = min(len(flat_ind1), len(flat_ind2))
    pos1, pos2 = [0] * size, [0] * size

    # Initialize the position of each index in the individuals
    for i in range(size):
        pos1[flat_ind1[i]] = i
        pos2[flat_ind2[i]] = i
    
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = flat_ind1[i]
        temp2 = flat_ind2[i]
        # Swap the matched value
        flat_ind1[i], flat_ind1[pos1[temp2]] = temp2, temp1
        flat_ind2[i], flat_ind2[pos2[temp1]] = temp1, temp2
        # Position bookkeeping
        pos1[temp1], pos1[temp2] = pos1[temp2], pos1[temp1]
        pos2[temp1], pos2[temp2] = pos2[temp2], pos2[temp1]

    # Increment node values back to original range
    flat_ind1 = [node + 1 for node in flat_ind1]
    flat_ind2 = [node + 1 for node in flat_ind2]
    

    ind1_reconstructed = reconstruct_individual(flat_ind1, route_lengths_1)
    ind2_reconstructed = reconstruct_individual(flat_ind2, route_lengths_2)
    
    return ind1_reconstructed, ind2_reconstructed

# mutation
def mut_shuffle_indexes(individual, indpb):
    """Shuffle the attributes of the input individual and return the mutant.
    The *indpb* argument is the probability of each attribute to be moved. 

    Args:
        individual: Individual to be mutated.
        indpb: Independent probability for each attribute to be exchanged to
                  another position.
        
    Returns:
        A tuple of one individual.
    """
    route_lengths = [len(route) for route in individual]
    flat_ind = [node for route in individual for node in route]
    
    size = len(flat_ind)
    for i in range(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            flat_ind[i], flat_ind[swap_indx] = flat_ind[swap_indx], flat_ind[i]

    individual = reconstruct_individual(flat_ind, route_lengths)
    return individual


def two_opt(route, distance_matrix):
    """
    Perform 2-opt local search on a given route to optimize it.
    
    Args:
        route: A list of integers representing the customers in the route.
        distance_matrix: A 2D list or numpy array containing the distances between nodes.
    
    Returns:
        A list of integers representing the optimized route.
    """
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                
                # Calculate the cost difference between the old route and the new route obtained by swapping edges
                old_cost = distance_matrix[route[i - 1]][route[i]] + distance_matrix[route[j]][route[j + 1]]
                new_cost = distance_matrix[route[i - 1]][route[j]] + distance_matrix[route[i]][route[j + 1]]
                if new_cost < old_cost:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
    return route

def simple_repair(initial_route, battery_capacity, energy_consumption, distance_matrix, station_list):
    """
    Repair the given route by ensuring the vehicle has enough energy to reach a charging station from each node.

    Args:
        initial_route (list): The initial route that needs to be repaired.
        battery_capacity (float): The capacity of the vehicle's battery.
        energy_consumption (float): The energy consumption per unit distance.
        distance_matrix (list of lists): A matrix representing the distances between each pair of nodes.
        station_list (list): A list of charging station indices.
    
    Returns:
        list: The repaired route, which ensures that the vehicle has enough energy to reach a charging station from each node.
    """
    route = initial_route.copy()

    if len(route) == 0 or len(route) == 1:
        return route
    repaired_route = [route.pop(0)]
    current_node = repaired_route[-1]
    next_node = route[0]

    # Calculate the remaining energy after departing from the first node
    energy_left = battery_capacity - energy_consumption * distance_matrix[0][current_node]
    
    while route:
        # Update the energy left after moving to the next node
        updated_energy_left = energy_left - energy_consumption * distance_matrix[current_node][next_node]
        nearest_station_to_next = find_nearest_station(next_node, distance_matrix, station_list)

        # Check if there is enough energy left to reach the nearest station from the next node
        if updated_energy_left >= energy_consumption * distance_matrix[next_node][nearest_station_to_next]:
            repaired_route.append(next_node)
            route.pop(0)
            next_node = route[0] if len(route) > 0 else 0
            current_node = repaired_route[-1]
            energy_left = updated_energy_left
        else:
            # Find stations reachable from the current node
            reachable_stations_from_current = [int(station) for station in station_list if energy_left >= energy_consumption * distance_matrix[current_node][int(station)]]
            nearest_station_to_next = find_nearest_station(next_node, distance_matrix, reachable_stations_from_current)

            repaired_route.append(nearest_station_to_next)
            energy_left = battery_capacity
            current_node = repaired_route[-1]

    # we have to check whether the energy left can support the vehicle to return back to the depot
    if energy_left < energy_consumption * distance_matrix[current_node][0]:
        nearest_station_to_current = find_nearest_station(current_node, distance_matrix, station_list)
        repaired_route.append(nearest_station_to_current)
    
    return repaired_route


def station_realloc_1(individual, battery_capacity, energy_consumption, distance_matrix, station_list):
    '''
    This function optimizes the route by relocating the position of the charging station in routes that only contain 
    one charging station. It attempts to insert the station between two gained station positions to improve the overall 
    performance of the route.
    
    Args:
        individual: A list of routes. Each route is a list of nodes.
        battery_capacity: The capacity of the battery.
        energy_consumption: The energy consumption rate of the vehicle.
        distance_matrix: A 2D list representing the distance between every two nodes.
        station_list: A list of charging stations.
        
    Returns:
        A list of optimized routes.
    '''
    def is_energy_left_no_less_than_zero(route, battery_capacity, energy_consumption, distance_matrix, station_list):
        # This helper function checks whether the energy left is no less than zero for a given route.
        prev_node = 0
        energy_left = battery_capacity
        for current_node in route[1:]:
            if str(current_node) in station_list:
                energy_left = battery_capacity
            else:
                energy_left -= energy_consumption * distance_matrix[prev_node][current_node]
                if energy_left < 0:
                    return False
            prev_node = current_node
        return True
    
    
    improved_individual = []
    for route in individual:
        real_route = [0] + route + [0]
        optimal_cost = sum(distance_matrix[real_route[i]][real_route[i + 1]] for i in range(len(real_route) - 1))
        optimal_route = route
        
        num_stations = sum(str(node) in station_list for node in route)
        if num_stations == 1:
            route_removed_stations = [node for node in real_route if str(node) not in station_list]
            reverse_route_removed_stations = route_removed_stations[::-1]

            route_forward  = [0] + simple_repair(route_removed_stations[1:-1], battery_capacity, energy_consumption, distance_matrix, station_list) + [0]
            index_station_in_forward_route = next(i for i, node in enumerate(route_forward) if str(node) in station_list)
            sublist_before_station_forward = route_forward[:index_station_in_forward_route] + [real_route[index_station_in_forward_route + 1]]

            route_backward = [0] + simple_repair(reverse_route_removed_stations[1:-1], battery_capacity, energy_consumption, distance_matrix, station_list) + [0]
            contains_station = any(str(node) in station_list for node in route_backward)
            if not contains_station:
                optimal_route = route_backward[1:-1]
                improved_individual.append(optimal_route)
                continue
            index_station_in_backward_route = next(i for i, node in enumerate(route_backward) if str(node) in station_list)
            sublist_before_station_backward = route_backward[:index_station_in_backward_route] + [route_backward[index_station_in_backward_route + 1]]

            # to get the valid edges between the two charging station
            intersected_list = []
            for node in sublist_before_station_forward:
                if node in sublist_before_station_backward:
                    intersected_list.append(node)
                    

            if len(intersected_list) <= 1:
                improved_individual.append(optimal_route)
                continue
                
            for idx, node in enumerate(intersected_list[:-1]):
                # try to insert station in the edge
                for station in station_list:
                    insert_point = route_removed_stations.index(node)            
                    temp_route = route_removed_stations[:insert_point + 1] + [int(station)] + route_removed_stations[insert_point + 1:]
                    temp_cost = sum(distance_matrix[temp_route[i]][temp_route[i + 1]] for i in range(len(temp_route) - 1))
                    if temp_cost < optimal_cost and is_energy_left_no_less_than_zero(temp_route, battery_capacity, energy_consumption, distance_matrix, station_list):
                        optimal_cost = temp_cost
                        optimal_route = temp_route[1:-1]
  
            reverse_intersected_list = intersected_list[::-1]
            for idx, node in enumerate(reverse_intersected_list[:-1]):
                # try to insert station in the edge
                for station in station_list:
                    insert_point = reverse_route_removed_stations.index(node)            
                    temp_route = reverse_route_removed_stations[:insert_point + 1] + [int(station)] + reverse_route_removed_stations[insert_point + 1:]
                    temp_cost = sum(distance_matrix[temp_route[i]][temp_route[i + 1]] for i in range(len(temp_route) - 1))
                    if temp_cost < optimal_cost and is_energy_left_no_less_than_zero(temp_route, battery_capacity, energy_consumption, distance_matrix, station_list):
                        optimal_cost = temp_cost
                        optimal_route = temp_route[1:-1]        
                        
        improved_individual.append(optimal_route)
        
    
    return improved_individual  



# ************************************************************************
# ******************************** GA ************************************

def local_search(individual, instance):
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
        extended_route = [0] + route + [0]
        improved_route = two_opt(extended_route, distance_matrix)
        optimized_individual.append(improved_route[1:-1])

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
    
def run_GA(instance, seed, pop_size, n_gen, cx_prob, mut_prob, indpb, result_dir, is_export_csv=True):
    
    random.seed(seed)
    
    CANDIDATES = []
    PLAIN_CANDIDATES_SET = []

    num_vehicles =  instance.num_of_vehicles
    num_customers = instance.dimension - 1
    capacity = instance.capacity
    demands = instance.demands
    station_list = instance.station_list
    
    csv_data = []
    
    # population initialization
    pop = []
    init_pop_randomly = [generate_individual_randomly(num_vehicles, num_customers) for _ in range(int(pop_size * 0.8))]
    init_pop_evenly = [generate_individual_evenly(num_vehicles, num_customers) for _ in range(int(pop_size * 0.2))]
    pop.extend(init_pop_randomly)
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
                    CANDIDATES.append((optimized_individual, cost))
        print(f'  Evaluated {stats_num_candidates_added} individuals')
        
        CANDIDATES.sort(key=lambda x: x[1])
        # Elites Population
        CANDIDATES = CANDIDATES[:1000]

        # Statistical Data
        size = len(CANDIDATES)
        fits = []
        mean = 0
        std  = 0.0
        min_fit = None
        max_fit = None

        if size == 0:
            print('  No candidates')
        else:
            fits = [fit for ind, fit in CANDIDATES]
            mean = sum(fits) / size
            min_fit = min(fits)
            max_fit = max(fits)

        if size > 1:
            std = (sum((x - mean) ** 2 for x in fits) / (size - 1)) ** 0.5
        else:
            std = 0.0
        print(f'  Candidates Num {size}')
        print(f'  Min {min_fit}') # the best result of each generation
        print(f'  Max {max_fit}')
        print(f'  Avg {mean}')   # Reflect the direction of population evolution 
        print(f'  Std {std}')
        
        min_individual = [] 
        min_fitness = None
        if size != 0:
            min_individual, min_fitness = CANDIDATES[0]
        print(f'  Best fitness: {min_fit}')
        best_solution = min_individual
        best_cost = min_fitness
        # Write data to holders for exporting results to CSV file
        if is_export_csv:
            csv_row = {
                'generation': gen,
                'evaluated_individuals': stats_num_candidates_added,
                'min_fitness': min_fit,
                'max_fitness': max_fit,
                'avg_fitness': mean,
                'std_fitness': std,
            }
            csv_data.append(csv_row)
    
        # select
        pop = []  
        elites = [ind for ind, fit in CANDIDATES]
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
        individuals_randomly = [generate_individual_randomly(num_vehicles, num_customers) for _ in range(int(num_left * 0.2))]
        individuals_from_immigration.extend(individuals_evenly)
        individuals_from_immigration.extend(individuals_randomly)
        
        pop.extend(individuals_from_pop_pool)
        pop.extend(individuals_from_immigration)

    print('------------End of (successful) evolution------------', end='\n\n') 

    csv_file = ''
    if is_export_csv:
        csv_file_name = f'{instance.name}_seed{seed}_nG{n_gen}.csv'
        csv_file = os.path.join(result_dir, csv_file_name) 
        print(f'Write to file: {csv_file}')
        make_dirs_for_file(path=csv_file)
        if not exist(path=csv_file, overwrite=True):
            with io.open(csv_file, 'wt', newline='') as file_object:
                fieldnames = [
                    'generation',
                    'evaluated_individuals',
                    'min_fitness',
                    'max_fitness',
                    'avg_fitness',
                    'std_fitness',
                ]
                writer = DictWriter(file_object, fieldnames=fieldnames, dialect='excel')
                writer.writeheader()
                for csv_row in csv_data:
                    writer.writerow(csv_row)
                    
    return (best_solution, best_cost, csv_file)
        

    
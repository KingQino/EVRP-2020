# -*- coding: utf-8 -*-
import os
import io
import time
import random
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from csv import DictWriter
from deap import base, creator, tools

from .evrp_instance import EvrpInstance 


def find_nearest_station(node_no, distance_matrix, station_list):
    '''Fine the nearest charging station of the given node
    
    Args:
        node: the index of the node
        distance_matrix: the distance matrix of nodes
        staion_list: the list of station nodes 
    
    Returns:
        the index of the station
    '''
    return int(min(station_list, key=lambda station_no: distance_matrix[int(node_no)][int(station_no)])) 

def is_feasible(route, capacity_left, energy_left, distance_matrix, energy_consumption, station_list):
    '''Judge whether the sub route is feasible, i.e., 
    1. The capacity left is greater than or equal to 0
    2. The state of battery (SOC) left is greater than or equal to 0
    3. The SOC of the vihicle can support the vehicle to return the depot or visit the nearest charging station
    
    Args:
        route: the sub-route
        capacity_left: the capacity left of the vehicle
        energy_left: the energy left of the vehicle
        distance_matrix: the distance matrix of nodes
        energy_consumption: the energy consumption rate
        staion_list: the list of station nodes 
        
    Returns:
        boolean value
    '''
    current = route[-1]
    nearest_station = find_nearest_station(current, distance_matrix, station_list)
    depot = 0

    # Calculate the energy required to reach the nearest station or the depot
    energy_required = energy_consumption * min(distance_matrix[current][nearest_station], distance_matrix[current][depot])
    
    if capacity_left >= 0 and energy_left >= 0 and energy_left >= energy_required:
        return True
    else:
        return False

def handle_infeasible_route(route, sub_route, nearest_station, updated_capacity_left, updated_energy_left, energy_left, prev_node, node, MAX_CAPACITY, BATTERY_CAPACITY, DISTANCE_MATRIX, ENERGY_CONSUMPTION, STATION_LIST):
    '''Handle the infeasible sub-route, namely, try to repair it or end it.
    
    Args:
        route: a list of sub_routes
        sub_route: a list of integers (i.e., customer and station nodes)
        nearest_station: the nearest charging station for the end node in the current sub-route
        updated_capacity_left: the capacity left of the vehicle after appending the current node to the sub-route 
        updated_energy_left: the energy left of the vehicle after appending the current node to the sub-route 
        energy_left: currently, the energy left of the vehicle (i.e., haven't appending the current node to the sub-route)
        prev_node: the previous customer node beforing appending the current node, i.e., the end node of current sub-route
        node: the current customer node, which is the node we try to append it to the sub-route
        MAX_CAPACITY: the max capacity of the EV, constant
        BATTERY_CAPACITY: The state of battery (SOC) left of the EV, constant
        DISTANCE_MATRIX: the distance matrix of nodes, constant
        ENERGY_CONSUMPTION: the energy consumption rate, constant
        STATION_LIST: the list of station nodes, constant
    
    Returns:
        a dictionary - {'route': route, 'sub_route': sub_route, 'capacity_left': capacity_left, 
        'energy_left': energy_left, 'prev_node': prev_node, 'assigned': assigned}
    '''
    assigned = False
    if updated_capacity_left < 0:
        # Handle the case where vehicle capacity is exhausted
        if energy_left >= ENERGY_CONSUMPTION * DISTANCE_MATRIX[prev_node][0]:
            route.append(sub_route)
        else:
            sub_route.append(nearest_station)
            route.append(sub_route)
    else:
        # Handle the case where vehicle energy is insufficient
        if energy_left >= ENERGY_CONSUMPTION * max(DISTANCE_MATRIX[prev_node][nearest_station], DISTANCE_MATRIX[prev_node][0]):
            # Starting from the prev_node, the EV has enough battery to reach the nearest charging station and return to the depot.
            sub_route.append(nearest_station)
            if is_feasible(sub_route + [node], updated_capacity_left, updated_energy_left, DISTANCE_MATRIX, ENERGY_CONSUMPTION, STATION_LIST):
                sub_route.append(node)
                capacity_left = updated_capacity_left
                energy_left = updated_energy_left
                prev_node = node
                assigned = True
                return {'route': route, 'sub_route': sub_route, 'capacity_left': capacity_left, 'energy_left': energy_left, 'prev_node': prev_node, 'assigned': assigned}
            else:
                sub_route.pop()
                route.append(sub_route)
        elif energy_left >= ENERGY_CONSUMPTION * DISTANCE_MATRIX[prev_node][nearest_station]:
            # Starting from the prev_node, the EV doesn't have enough battery to return to the depot, but can reach the nearest charging station.
            sub_route.append(nearest_station)
            if is_feasible(sub_route + [node], updated_capacity_left, updated_energy_left, DISTANCE_MATRIX, ENERGY_CONSUMPTION, STATION_LIST):
                sub_route.append(node)
                capacity_left = updated_capacity_left
                energy_left = updated_energy_left
                prev_node = node
                assigned = True
                return {'route': route, 'sub_route': sub_route, 'capacity_left': capacity_left, 'energy_left': energy_left, 'prev_node': prev_node, 'assigned': assigned}
            else:
                route.append(sub_route)
        else:
            # Starting from the prev_node, the EV cannot reach the nearest station or return to the depot
            route.append(sub_route)
    
    sub_route = []
    capacity_left = MAX_CAPACITY
    energy_left = BATTERY_CAPACITY
    prev_node = 0
    
    return {'route': route, 'sub_route': sub_route, 'capacity_left': capacity_left, 'energy_left': energy_left, 'prev_node': prev_node, 'assigned': assigned}

def individual_2_route(individual, instance):
    '''Convert the individual (encoded as a list of integers) into a list of feasible routes.
    The individual is encoded as a list of customer nodes, and each feasible route is preceded by a vehicle.
    
    Args:
        individual: a list of integers (customer nodes)
        instance: the benchmark instance got from the original data
        
    Returns:
        routes - a list of feasible sub-routes (each sub-route might contain customer or station nodes)
    '''
    route = []

    # Retrieve necessary instance parameters
    MAX_CAPACITY = instance.capacity
    BATTERY_CAPACITY = instance.battery_capacity
    ENERGY_CONSUMPTION = instance.energy_consumption
    STATION_LIST = instance.station_list
    DISTANCE_MATRIX = instance.distance_matrix
    DEMANDS = instance.demands

    # Initialize a sub-route and related variables
    sub_route = []
    capacity_left = MAX_CAPACITY
    energy_left = BATTERY_CAPACITY
    prev_node = 0

    # Iterate through customer nodes and insert them into feasible sub-routes
    for node in individual:
        assigned = False
        while not assigned:
            # Calculate remaining capacity and energy after inserting the current customer node
            updated_capacity_left = capacity_left - DEMANDS[f'{node}']
            updated_energy_left = energy_left - ENERGY_CONSUMPTION * DISTANCE_MATRIX[prev_node][node]

            # Check if the updated route is feasible
            if is_feasible(sub_route + [node], updated_capacity_left, updated_energy_left, DISTANCE_MATRIX, ENERGY_CONSUMPTION, STATION_LIST):
                # Add the customer node to the sub-route and update related variables
                sub_route.append(node)
                capacity_left = updated_capacity_left
                energy_left = updated_energy_left
                prev_node = node
                assigned = True
            else:
                # Determine the reason for infeasibility and take appropriate action
                if(prev_node == 0):
                    nearest_station = find_nearest_station(node, DISTANCE_MATRIX, STATION_LIST)
                    sub_route.append(nearest_station)
                    prev_node = nearest_station
                    continue
                nearest_station = find_nearest_station(prev_node, DISTANCE_MATRIX, STATION_LIST)
                updated_energy_left = BATTERY_CAPACITY - ENERGY_CONSUMPTION * DISTANCE_MATRIX[nearest_station][node]

                result = handle_infeasible_route(route, sub_route, nearest_station, updated_capacity_left, 
                                                 updated_energy_left, energy_left, prev_node, node, MAX_CAPACITY, 
                                                 BATTERY_CAPACITY, DISTANCE_MATRIX, ENERGY_CONSUMPTION, STATION_LIST)
                route, sub_route, capacity_left, energy_left, prev_node, assigned = result.values()

    # Handle the remaining sub_route, if any
    if sub_route != []:
        if energy_left >= ENERGY_CONSUMPTION * DISTANCE_MATRIX[sub_route[-1]][0]:
            route.append(sub_route)
        else:                            
            nearest_station = find_nearest_station(sub_route[-1], DISTANCE_MATRIX, STATION_LIST)
            sub_route.append(nearest_station)
            route.append(sub_route)

    return route

def plot_route(route, df, ax, route_color='green', linewidth=1):
    # Add the depot index (0) at the beginning and end of the route
    route_with_depot = [0] + route + [0]

    for i in range(len(route_with_depot) - 1):
        start_customer_idx = route_with_depot[i]
        end_customer_idx = route_with_depot[i + 1]

        x1, y1 = df['x_pos'].loc[start_customer_idx], df['y_pos'].loc[start_customer_idx]
        x2, y2 = df['x_pos'].loc[end_customer_idx], df['y_pos'].loc[end_customer_idx]

        ax.plot([x1, x2], [y1, y2], color=route_color, linewidth=linewidth)

def visualize_routes(routes, df, title='Route Plot'):
    colors = {'depot': 'red', 'customer': 'blue', 'station': 'black'}
    markers = {'depot': 'D', 'customer': 'o', 'station': 's'}

    fig, ax = plt.subplots(figsize=(12, 8))
    for label, group in df.groupby('label'):
        if label == 'depot':
            group.plot(kind='scatter', x='x_pos', y='y_pos', label=label, color=colors[label], marker=markers[label], ax=ax, s=30)
        elif label == 'customer':
            group.plot(kind='scatter', x='x_pos', y='y_pos', label=label, color=colors[label], marker=markers[label], ax=ax)
        else:  # For the 'station' label
            group.plot(kind='scatter', x='x_pos', y='y_pos', label=label, color=colors[label], marker=markers[label], ax=ax, s=30) 
            

        # Add node_no labels for customer nodes
        if label == 'customer':
            for index, row in group.iterrows():
                ax.text(row['x_pos'] + 1, row['y_pos'] - 1, str(row['node_no']), fontsize=10, color=colors[label])
        elif label == 'station':
            for index, row in group.iterrows():
                ax.text(row['x_pos'] + 1, row['y_pos'] + 1, str(row['node_no']), fontsize=10, color=colors[label])
    
    # Create a colormap and generate a list of colors for each route
    colormap = plt.cm.get_cmap('tab10', len(routes))
    colors = [mcolors.rgb2hex(colormap(i)[:3]) for i in range(len(routes))]

    for i, route in enumerate(routes):
        plot_route(route, df, ax, route_color=colors[i])
    
    
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Move the legend and show the plot inside the loop
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def fitness_evaluation(individual, instance):
    '''Evaluate the generated routes
    
    Args:
        individual: a list of integers and each integer denote a customer
        instance: the benchmark instance got from the original data
        
    Returns:
        tuple - (float, ) single objective fitness, which is used to satisfy the requirement of `deap.creator` fitness
    '''
    routes = individual_2_route(individual, instance)
    
    
    total_distance = 0.0
    DISTANCE_MATRIX = instance.distance_matrix

    
    for route in routes:
        _route = [0] + route + [0]
        for current in range(1, len(_route)):
            prev_node = _route[current - 1]
            cur_node  = _route[current]
            total_distance += DISTANCE_MATRIX[prev_node][cur_node]
    
    return (total_distance, )

def create_dataframe(instance):
    '''Create a dataframe from the instance

    Args:
        instance: the Object obtained from the Benchmark Instance
    '''
    # Create an empty dataframe
    df = pd.DataFrame(columns=['node_no', 'x_pos', 'y_pos', 'label'], dtype=str)

    # Iterate through key-value pairs, unpacking the tuple in the loop
    for key, value in instance.node_coordinates.items():
        if key == instance.depot_index:
            new_row = pd.DataFrame({'node_no': [str(key)],
                                    'x_pos': [value[0]],
                                    'y_pos': [value[1]],
                                    'label': ['depot']
                                   })
            df = pd.concat([df, new_row], ignore_index=True)        
        elif key in instance.station_list:
            new_row = pd.DataFrame({'node_no': [str(key)],
                                    'x_pos': [value[0]],
                                    'y_pos': [value[1]],
                                    'label': ['station']
                                   })
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            new_row = pd.DataFrame({'node_no': [str(key)],
                                    'x_pos': [value[0]],
                                    'y_pos': [value[1]],
                                    'label': ['customer']
                                   })
            df = pd.concat([df, new_row], ignore_index=True)
    
    return df

def generate_individual(num_of_customers):
    '''Generate individual randomly
    
    Args:
        num_of_customers: the number of customers
    
    Returns:
        individual - a list of integers (i.e., customers) 
    '''
    individual = list(range(1, num_of_customers))
    random.shuffle(individual)

    return individual

def init_population(seed, pop_size, ind_size):
    '''Initialize the population
    
    Args:
        seed: random seed, which makes the results reproductive
        pop_size: the population size
        ind_size: the individual size, but acctually the generated individual length is ind_size - 1
        
    Returns:
        a list of individual
    '''
    random.seed(seed)
    population = []
    for _ in range(pop_size):
        individual = generate_individual(ind_size)
        population.append(individual)
    return population

def cx_partialy_matched(ind1, ind2):
    '''Perform a partially matched crossover (PMX) on the given individuals.
    
    This function is a modified version of the `deap.tools.cxPartialyMatched` function
    that supports individual lists that start the `indexes` from 1 rather than 0.
    
    Args:
        ind1: A list of integers representing the first parent.
        ind2: A list of integers representing the second parent.
    
    Returns:
        A tuple of two variable pointers (`ind1` and `ind2`), which also means the two input arguments have been changed.
    '''
    # Convert individual lists to start with index 0
    ind_1 = [x - 1 for x in ind1]
    ind_2 = [x - 1 for x in ind2]
    
    # Perform crossover using the modified individual lists
    tmp_1, tmp_2 = tools.cxPartialyMatched(ind_1, ind_2)

    # Convert offspring lists back to start with index 1
    ans_1 = [x + 1 for x in tmp_1]
    ans_2 = [x + 1 for x in tmp_2]
    
    size =  min(len(ind1), len(ind2))
    for i in range(size):
        ind1[i] = ans_1[i]
        ind2[i] = ans_2[i]
    
    # Return the offspring
    return ind1, ind2


def selection(population, size, rate=0.2):
    '''This is a customized selection function. It will select a certain-size subset from the populaltion,
    and the subset has a proportion of elites.

    Args:
        population: A list of individuals, each of which is a list of values.
        size: the size of the selected subset from the population.
        rate: the proportion of elites.
    
    Returns:
        The selected subset of the population.
    '''
    top_size = int(size * rate)
    top_individuals = tools.selBest(population, top_size)  
    
    remaining_size = size - top_size
    remaining_individuals = tools.selRoulette(population, remaining_size) 
    
    selected_pop = top_individuals + remaining_individuals
    
    return selected_pop[:size]


def deduplicate_population(population):
    """Deduplicate a population by removing identical individuals.
    
    Args:
        population: A list of individuals, each of which is a list of values.
    
    Returns:
        A list of deduplicated individuals.
    """
    population.sort()
    length = len(population)
    lastItem = population[length - 1]
    for i in range(length - 2,-1,-1):
            currentItem = population[i]
            if currentItem == lastItem:
                    population.remove(currentItem)
            else:
                    lastItem = currentItem
    
    return population

def make_dirs_for_file(path):
    '''Make directories for the file
    
    Args:
        path: the given file path
    '''
    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass

def guess_path_type(path):
    '''judge the type of the given path
    
    Args:
        path: the given file path
    '''
    if os.path.isfile(path):
        return 'File'
    if os.path.isdir(path):
        return 'Directory'
    if os.path.islink(path):
        return 'Symbolic Link'
    if os.path.ismount(path):
        return 'Mount Point'
    return 'Path'

def exist(path, overwrite=False, display_info=False):
    '''judge whether it exists for the given path
    
    Args:
        path: the given file path
        overwrite: whether overwrite the file
        display_info: whether display the info of the path
    '''
    if os.path.exists(path):
        if overwrite:
            if display_info:
                print(f'{guess_path_type(path)}: {path} exists. Overwrite.')
            os.remove(path)
            return False
        if display_info:
            print(f'{guess_path_type(path)}: {path} exists.')
        return True
    if display_info:
        print(f'{guess_path_type(path)}: {path} does not exist.')
    return False

def plot_training_graph(df, title, is_save=False, save_path='./results/instance/training.png'):
    """Plot a line graph for avg_fitness and min_fitness by generation using a DataFrame.
    It's used to visualize the training process.
    
    Args:
        df: A pandas DataFrame containing 'generation', 'avg_fitness', and 'min_fitness' columns.
        title: the graph title.
        is_save: whether save the image.
        save_path: the save path.
    """
    # Set the figure size (width, height) in inches
    plt.figure(figsize=(12, 6))

    # Plot the line graph for avg_fitness
    plt.plot(df['generation'], df['avg_fitness'], label='Average Fitness')

    # Plot the line graph for min_fitness
    plt.plot(df['generation'], df['min_fitness'], label='Minimum Fitness')

    plt.plot(df['generation'], df['max_fitness'], label='Maximum Fitness')

    # Set the labels for the X and Y axes
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    # Set the title for the graph
    plt.title(f'{title} Training Process')

    # Add a legend to the graph
    plt.legend()

    if is_save:
        plt.savefig(save_path)

    # Display the graph
    plt.show()
 


def elapsed_time(start):
    """
    Calculate the elapsed time since the 'start' timestamp.

    Args:
        start: A float representing the start time, typically obtained
            by calling time.process_time() before the operation to be timed.

    Returns:
        A float representing the elapsed time in seconds.
    """
    return time.process_time() - start

def run_ga(instance_name, seed, pop_size, n_gen, rate_elite, cx_prob, mut_prob, immig_prop, export_csv=True):
    
    random.seed(seed)
    
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname('.')))
    DATA_DIR = os.path.join(BASE_DIR, 'evrp-benchmark-set')
    file_dir = os.path.join(DATA_DIR, instance_name)
    
    instance = EvrpInstance(file_dir)
    
    IND_SIZE = instance.dimension - 1
    
    creator.create('FitnessMin', base.Fitness, weights=(-1.0, ))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    
    
    # -------- In `toolbox`, register regarding functions --------
    
    # individual and population initialization approach
    toolbox.register('indexes', random.sample, range(1, IND_SIZE + 1), IND_SIZE)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual, n=pop_size)
    # evaluation
    toolbox.register('evaluate', fitness_evaluation, instance=instance)
    # selection
    toolbox.register('select', selection, rate=rate_elite)
    # mate (crossover)
    toolbox.register('mate', cx_partialy_matched)
    # mutation
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb=mut_prob)
    
    
    csv_data = []
    print('----------------Start of evolution----------------')
    start_time = time.process_time()

    # Initialize population
    pop = toolbox.population()
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print(f'  Evaluated {len(pop)} individuals')
    
    
    # Begin the evolution
    for gen in range(n_gen):
        print(f'---- Generation {gen} ----')
        
        
        # Clone the population, and the individuals produce offspring through crossover and mutation
        offspring = list(map(toolbox.clone, pop))
        # Apply crossover and mutation on the offspring
        # ---------------  Crossover  -----------------
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # ---------------  Mutation  -----------------
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'  Evaluated {len(invalid_ind)} individuals')

        # ---------------  Immigration  -----------------
        immigration = toolbox.population(n=int(pop_size * immig_prop))
        fitnesses = map(toolbox.evaluate, immigration)
        for ind, fit in zip(immigration, fitnesses):
            ind.fitness.values = fit
        
        # ---------------  Select  -----------------
        # Select the specific number of individuals from the current population pool
        selected_pop = toolbox.select(offspring + pop + immigration, pop_size)
        

        # The population is entirely replaced by the offspring
        pop[:] = selected_pop
        
        # Collect the statistical info of each generation
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum([x**2 for x in fits])
        std = abs(sum2 / length - mean**2)**0.5
        print(f'  Min {min(fits)}') # the best result of each generation
        print(f'  Max {max(fits)}')
        print(f'  Avg {mean}')   # Reflect the direction of population evolution 
        print(f'  Std {std}')
        # Write data to holders for exporting results to CSV file
        if export_csv:
            csv_row = {
                'generation': gen,
                'evaluated_individuals': len(invalid_ind),
                'min_fitness': min(fits),
                'max_fitness': max(fits),
                'avg_fitness': mean,
                'std_fitness': std,
                'duration': elapsed_time(start_time)
            }
            csv_data.append(csv_row)
    
    print('------------End of (successful) evolution------------', end='\n\n')
    duration = elapsed_time(start_time)
    print(f'The Total Evolutionary Duration: {duration} for {n_gen} generations')
    print(f'The average evolutionary time for each generation is: {duration/n_gen}')

    
    
    best_ind = tools.selBest(pop, 1)[0]
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {best_ind.fitness.values[0]}')
    routes = individual_2_route(best_ind, instance)
    for idx, route in enumerate(routes):
        _route = [0] + route + [0]
        print(f' vehicle {idx}: ', _route)
    print(f'Total cost: {best_ind.fitness.values[0]}')
    csv_file = ''
    if export_csv:
        csv_file_name = f'{instance_name}_seed{seed}_iS{IND_SIZE}_pS{pop_size}_rE{rate_elite}_cP{cx_prob}_mP{mut_prob}_iP{immig_prop}_nG{n_gen}.csv'
        csv_file = os.path.join(BASE_DIR, 'results/encoding-without-vehicle-assignment-info', csv_file_name)
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
                    'duration',
                ]
                writer = DictWriter(file_object, fieldnames=fieldnames, dialect='excel')
                writer.writeheader()
                for csv_row in csv_data:
                    writer.writerow(csv_row)
    
    return (best_ind, routes, best_ind.fitness.values[0], csv_file)        


# -*- coding: utf-8 -*-
import os
import time
import pandas as pd


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



# ************************************************************************
# *********************** Feasibility Judgement **************************

def is_capacity_feasible(individual, capacity, demands):
    """Check if a given individual is feasible with respect to vehicle capacity.
    
    Args:
        individual: A list of lists, where each sublist represents a vehicle and its assigned customers.
        capacity: The capacity of the vehicles.
        demands: The dictionary containing the demands of each customer.
        
    Returns:
        bool: True if the individual is feasible, False otherwise.
    """
    
    # Iterate through each route in the individual
    for route in individual:
        # Calculate the total demand of the customers in the route
        total_demand = sum(demands[f'{customer}'] for customer in route)
        
        # If the total demand exceeds the vehicle's capacity, the individual is not feasible
        if total_demand > capacity:
            return False
    
    # If none of the routes exceed the vehicle's capacity, the individual is feasible
    return True




# ************************************************************************
# ***************************** Auxiliary ********************************

def reconstruct_individual(flat_ind, route_lengths):
    '''Reconstruct the original format of an individual from a flattened version.
    
    Args:
        flat_ind: A flattened version of an individual, where all routes are combined into a single list.
        route_lengths: A list of integers representing the length of each route in the original individual format.

    Returns:
        A list of lists, where each sublist represents a vehicle and its assigned customers.
    '''
    individual = []
    start_idx = 0
    for length in route_lengths:
        individual.append(flat_ind[start_idx:start_idx + length])
        start_idx += length
    return individual


def deduplicate_population(population):
    """
    Remove duplicate individuals from a population.
    
    Args:
        population: A list of individuals.
        
    Returns:
        A deduplicated population as a list.
    """
    deduplicated_population = []

    for individual in population:
        if individual not in deduplicated_population:
            deduplicated_population.append(individual)

    return deduplicated_population

def find_nearest_station(node_no, distance_matrix, station_list):
    '''Find the nearest charging station of the given node
    
    Args:
        node: the index of the node
        distance_matrix: the distance matrix of nodes
        staion_list: the list of station nodes 
    
    Returns:
        the index of the station
    '''
    return int(min(station_list, key=lambda station_no: distance_matrix[int(node_no)][int(station_no)])) 


def fitness_evaluation(individual, distance_matrix):
    '''Evaluate the generated routes
    
    Args:
        individual: A list of routes. Each route is a list of nodes.
        instance: the benchmark instance got from the original data
        
    Returns:
        tuple - (float, ) single objective fitness, which is used to satisfy the requirement of `deap.creator` fitness
    '''    
    
    total_distance = 0.0
    
    for route in individual:
        _route = [0] + route + [0]
        for current in range(1, len(_route)):
            prev_node = _route[current - 1]
            cur_node  = _route[current]
            total_distance += distance_matrix[prev_node][cur_node]
    
    return total_distance



# ************************************************************************
# ******************************* Plot ***********************************

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

def plot_nodes(df, title='Scatter Plot', is_save=False, save_path='./results/pictures/scatter_plot.png'):
    '''Plot the instance from the benchmark

    Args:
        df: pandas dataframe, it contains "node_no	x_pos	y_pos	label" columns
        title: the title of the plot
        save_path: the save path of the plot
    '''
    colors = {'depot': 'red', 'customer': 'blue', 'station': 'black'}
    markers = {'depot': 'D', 'customer': 'o', 'station': 's'}

    fig, ax = plt.subplots(figsize=(12, 8))
    for label, group in df.groupby('label'):
        if label == 'depot':
            group.plot(kind='scatter', x='x_pos', y='y_pos', label=label, color=colors[label], marker=markers[label], ax=ax)
        else:
            group.plot(kind='scatter', x='x_pos', y='y_pos', label=label, color=colors[label], marker=markers[label], ax=ax)

        # Add node_no labels for customer nodes
        if label == 'customer':
            for index, row in group.iterrows():
                ax.text(row['x_pos'] + 1, row['y_pos'] - 1, str(row['node_no']), fontsize=10, color=colors[label])
        elif label == 'station':
            for index, row in group.iterrows():
                ax.text(row['x_pos'] + 1, row['y_pos'] + 1, str(row['node_no']), fontsize=10, color=colors[label])

    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Move the legend and show the plot inside the loop
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot to the specified path
    if is_save:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def plot_route(route, df, ax, route_color='green', linewidth=1):
    '''Plots a single route on a given Axes instance.

    Args:
        route: A list of nodes representing the route.
        df: A DataFrame with node details, with columns 'x_pos' and 'y_pos' for coordinates.
        ax: The Axes instance on which the route is to be plotted.
        route_color: The color to be used for the route. Defaults to 'green'.
        linewidth: The width of the line representing the route. Defaults to 1.
    '''
    # Add the depot index (0) at the beginning and end of the route
    route_with_depot = [0] + route + [0]

    for i in range(len(route_with_depot) - 1):
        start_customer_idx = route_with_depot[i]
        end_customer_idx = route_with_depot[i + 1]

        x1, y1 = df['x_pos'].loc[start_customer_idx], df['y_pos'].loc[start_customer_idx]
        x2, y2 = df['x_pos'].loc[end_customer_idx], df['y_pos'].loc[end_customer_idx]

        ax.plot([x1, x2], [y1, y2], color=route_color, linewidth=linewidth)

def visualize_routes(routes, df, title='Routes', is_save=False, save_path='./results/instance/routes.png'):
    '''Visualizes all routes on a single plot.

    Args:
        routes: A list of routes, where each route is a list of nodes.
        df: A DataFrame with node details, with columns 'x_pos' and 'y_pos' for coordinates.
        title: The title of the plot. Defaults to 'Route Plot'.
        is_save: whether save the image.
        save_path: the save path.
    '''
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
    if is_save:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

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
 


# ************************************************************************
# ***************************** Validate *********************************

class InfeasibleError(Exception):
    pass

def validate_individual_details(individual, instance, is_detailed=False):
    '''Validate the results of the generated individual, check the feasibility of them.
    The function would check each route, if the route is infeasible raise the InfeasibleError.
    
    Args:
        individual: the individual after simple repair, and they should be feasible 
            without energy consumption constraint concern
        instance: the benchmark instance got from the original data
        is_detailed: whether print the detailed info
    '''

    MAX_CAPACITY = instance.capacity
    BATTERY_CAPACITY = instance.battery_capacity
    ENERGY_CONSUMPTION = instance.energy_consumption
    STATION_LIST = instance.station_list
    DISTANCE_MATRIX = instance.distance_matrix
    DEMANDS = instance.demands

    delim = '-'
    
    if is_detailed:
        print(individual)
    
    for idx, route in enumerate(individual):
        
        real_route = [0] + route + [0]
        
        capacity_left = MAX_CAPACITY
        energy_left   = BATTERY_CAPACITY
        prev_node = 0
        
        detailed_info = []

        for node in real_route:
            if str(node) in STATION_LIST:
                energy_left = BATTERY_CAPACITY
            else:
                capacity_left -= DEMANDS[f'{node}']
                energy_left -= ENERGY_CONSUMPTION * DISTANCE_MATRIX[prev_node][node]
            
            detailed_info.append('{}({}, {:.2f})'.format(node, capacity_left, energy_left))
            
            if capacity_left < 0 or energy_left < 0:
                print('Infeasible route: ')
                print(delim.join(detailed_info))
                raise InfeasibleError('The route is infeasible')
            
            prev_node = node

        if is_detailed:
            print(delim.join(detailed_info), end='\n\n')



# ************************************************************************
# ***************************** Save File ********************************

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


# ************************************************************************
# ***************************** Statistics *******************************

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


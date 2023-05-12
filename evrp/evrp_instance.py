# -*- coding: utf-8 -*-

import os
import numpy as np

class EvrpInstance:
    def __init__(self, filename):
        self.filename = filename
        
        # the name of the instance 
        self.name = None
        
        # the variables below can be read directly from the 
        self.optimum = None # the optimum value
        self.num_of_vehicles = None
        self.dimension = None
        self.num_of_stations = None # the number of recharging stations 
        self.capacity = None # the maximum capacity of vehicle
        self.battery_capacity = None # the battery capacity of EV
        self.energy_consumption = None # the energy consumption rate
        self.edge_weight_format = None
        self.node_coordinates = {} # dictionary {'0': [145, 215], ...}
        self.demands = {} # {'0': 0, '1':1100}
        self.station_list = [] # ['22', '23', '24'] 
        self.depot_index = None # 1
        
        # the variables below can be gotten after processing
        self.actual_problem_size = None # Total number of customers, charging stations and depot
        self.distance_matrix = None
        
        self.read_problem()

    def read_problem(self):   
        self.name = os.path.splitext(self.filename.split('/')[-1])[0]
        with open(self.filename, 'rt', encoding='utf-8', newline='') as f:
            line = f.readline().strip()
            while line:
                if line.startswith('OPTIMAL_VALUE:'):
                    self.optimum = float(line.split()[-1])
                elif line.startswith('VEHICLES:'):
                    self.num_of_vehicles = int(line.split()[-1])
                elif line.startswith('DIMENSION:'):
                    self.dimension = int(line.split()[-1])
                elif line.startswith('STATIONS:'):
                    self.num_of_stations = int(line.split()[-1])
                elif line.startswith('CAPACITY:'):
                    self.capacity = int(line.split()[-1])
                elif line.startswith('ENERGY_CAPACITY:'):
                    self.battery_capacity = int(line.split()[-1])       
                elif line.startswith('ENERGY_CONSUMPTION:'):
                    self.energy_consumption = float(line.split()[-1])
                elif line.startswith('EDGE_WEIGHT_FORMAT:'):
                    self.edge_weight_format = line.split()[-1]                 

                elif line.startswith('NODE_COORD_SECTION'):
                    for i in range(self.dimension + self.num_of_stations):
                        data = f.readline().split()
                        if len(data) == 3:
                            self.node_coordinates[str(int(data[0]) - 1)] = [int(data[1]), int(data[2])]
                        else:
                            raise Exception("Invalid coordanate section")
                elif line.startswith('DEMAND_SECTION'):
                    for i in range(self.dimension):
                        data = f.readline().split()
                        if len(data) == 2:
                            self.demands[str(int(data[0]) - 1)] = int(data[1])
                        else:
                            raise Exception("Invalid demand section")            
                elif line.startswith('STATIONS_COORD_SECTION'):
                    for i in range(self.num_of_stations):
                        self.station_list.append(str(int(f.readline().split()[0]) - 1))    
                elif line.startswith('DEPOT_SECTION'):
                    self.depot_index = str(int(f.readline().strip()) - 1)
                line = f.readline().strip() 
        
        # make some process the data
        self.actual_problem_size = self.dimension + self.num_of_stations
        
        if self.edge_weight_format == 'EUC_2D':
            node_coords = np.array(list(self.node_coordinates.values()))
            self.distance_matrix = np.sqrt(np.sum((node_coords[:,np.newaxis] - node_coords)**2, axis=2))            
        else:
            raise ValueError("Unsupported edge weight format: {}".format(self.edge_weight_format))

    def get_distance(self, node_1, node_2):
        '''Get the Euclidean distance between two nodes.

        Args:
            node_1: a index of a node
            node_2: a index of a node

        Returns:
            a real number denoting the distance  
        '''
        return self.distance_matrix[int(node_1)][int(node_2)]
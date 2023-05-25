# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd

from evrp.evrp_instance import EvrpInstance
from evrp.encoding_02 import run_GA, run_large_instance, create_dataframe 
from evrp.utils import plot_training_graph, visualize_routes

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname('.')))
DATA_DIR = os.path.join(BASE_DIR, 'evrp-benchmark-set')
RESULT_DIR = os.path.join(BASE_DIR, 'results/encoding-with-vehicle-assignment-info/')

def main():
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--instance_name','-n',type=str, default = "E-n22-k4.evrp",required=True,help="a instance name")
    parser.add_argument('--seed','-s',type=int, default=1,help='random seed')
    parser.add_argument('--pop_size','-ps',type=int, default=10000,help='population size')
    parser.add_argument('--n_gen','-g',type=int, default=150,help='the training generation number')
    # parser.add_argument('--rate_elite','-r',type=float, default=0.2,help='The rate of elites for the population when selecting')
    parser.add_argument('--cx_prob','-c',type=float, default=0.8,help='the probability of crossover')
    parser.add_argument('--mut_prob','-m',type=float, default=0.5,help='the probability of mutation')
    parser.add_argument('--indpb','-ind',type=float, default=0.2,help='the mutation probability for each element in the sequence')
    # parser.add_argument('--immig_prop','-i',type=float, default=0.2,help='the proportion of immigrants in the new population')

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

    best_solution = []
    best_cost = None
    training_file = ''
    if instance.name.startswith('E'):
        best_solution, best_cost, training_file = run_GA(instance, seed, pop_size, n_gen, cx_prob, mut_prob, indpb, RESULT_DIR)
    elif instance.name.startswith('X'):
        best_solution, best_cost, training_file = run_large_instance(instance, seed, pop_size, n_gen, cx_prob, mut_prob, indpb, RESULT_DIR)
    else:
        raise NameError("The input argument '-n', namely, '--instance_name' is incorrect!")

    df = create_dataframe(instance)


    print(f'Excepted optimum: {instance.optimum}')
    print(f'The optimal result cost from my GA: {best_cost}')
    print(best_solution)

    visualize(training_file)
    visualize_routes(best_solution, df, 'Encoding Scheme 02 Best Solution for ' + instance.name, is_show=False, is_save=True, save_path=RESULT_DIR + instance.name.split('.')[0] + '-routes.png')



def visualize(training_file):
    df_run = pd.read_csv(training_file)
    instance_name = os.path.basename(training_file).split('_')[0].split('.')[0] 
    title = instance_name + ' | GA Encoding with Vehicle Assignment Info'
    plot_training_graph(df_run, title, is_show=False, is_save=True, save_path=RESULT_DIR + instance_name + '-evolution.png')

if __name__ == '__main__':
    main()
    # visualize("/Users/yhq/Desktop/Code/EVRP-2020/results/E-n22-k4.evrp_seed1_iS21_pS1000_rE0.2_cP0.8_mP0.5_iP0.2_nG600.csv")
    

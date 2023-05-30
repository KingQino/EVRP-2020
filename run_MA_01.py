# -*- coding: utf-8 -*-

import os
import argparse


from evrp.evrp_instance import EvrpInstance
from evrp.utils import *
from evrp.ma_01 import run_MA


BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname('.')))
DATA_DIR = os.path.join(BASE_DIR, 'evrp-benchmark-set')
RESULT_DIR = os.path.join(BASE_DIR, 'results/ma-01/')

def main():
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--instance_name','-n',type=str, default = "E-n22-k4.evrp",required=True,help="a instance name")
    parser.add_argument('--seed','-s',type=int, default=1,help='random seed')
    parser.add_argument('--pop_size','-ps',type=int, default=10000,help='population size')
    parser.add_argument('--n_gen','-g',type=int, default=150,help='the training generation number')
    parser.add_argument('--cx_prob','-c',type=float, default=0.8,help='the probability of crossover')
    parser.add_argument('--mut_prob','-m',type=float, default=0.5,help='the probability of mutation')
    parser.add_argument('--indpb','-ind',type=float, default=0.2,help='the mutation probability for each element in the sequence')
    parser.add_argument('--candidates_ratio','-cr',type=float, default=0.5,help='the preset size of the candite pool, a ratio of pop pool')
    parser.add_argument('--random_rate','-rr',type=float, default=0.8,help='for the population initialization, the ratio of generating population by the random way')
    parser.add_argument('--ng_pop_rate','-ngpr',type=float, default=0.7,help='for the next generation, the rate from the poplation pool after selecting the elites')
    parser.add_argument('--ng_immig_rate','-ngir',type=float, default=0.3,help='for the next generation, the rate from the immigrants after selecting the elites')


    args = parser.parse_args()

    instance_name = args.instance_name 
    seed = args.seed
    pop_size = args.pop_size
    n_gen = args.n_gen
    cx_prob = args.cx_prob
    mut_prob = args.mut_prob
    indpb = args.indpb
    cand_ratio = args.candidates_ratio
    rand_rate = args.random_rate
    ng_pop_rate = args.ng_pop_rate
    ng_immig_rate = args.ng_immig_rate


    file_dir = os.path.join(DATA_DIR, instance_name)
    instance = EvrpInstance(file_dir)

    best_solution, best_cost, training_file = run_MA(instance, seed, pop_size, n_gen, cx_prob, mut_prob, indpb, cand_ratio, rand_rate, ng_pop_rate, ng_immig_rate, RESULT_DIR)

    df = create_dataframe(instance)


    print(f'Excepted optimum: {instance.optimum}')
    print(f'The optimal result cost from my GA: {best_cost}')
    print(best_solution)

    visualize(training_file)
    visualize_routes(best_solution, df, 'MA wiht only simple-repair operator Best Solution for ' + instance.name, is_show=False, is_save=True, save_path=RESULT_DIR + instance.name.split('.')[0] + '-routes.png')



def visualize(training_file):
    df_run = pd.read_csv(training_file)
    instance_name = os.path.basename(training_file).split('_')[0].split('.')[0] 
    title = instance_name + ' | MA with only simple-repair local search operator'
    plot_training_graph(df_run, title, is_show=False, is_save=True, save_path=RESULT_DIR + instance_name + '-evolution.png')


if __name__ == '__main__':
    main()
    # visualize("/Users/yhq/Desktop/Code/EVRP-2020/results/E-n22-k4_seed1_nG200.csv")
    

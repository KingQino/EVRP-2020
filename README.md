# EVRP-2020

* [Benchmark dataset](https://mavrovouniotis.github.io/EVRPcompetition2020/)
  - some bugs in the benchmark
    - The instance `E-n30-k3.evrp` is a wrong named instance because it has 4 vhicles, so we should change the name of it to `E-n30-k4.evrp`
    - the results from the heuristic approaches are better than that obtained by the exact algorothms, which is really confusing.
* Approaches
  - [x] Simple Genetic Algorithm
    - Improvement is needed
    - encoding scheme & operators
    - [ ] for the first instance `E-n22-k4.evrp`, there is still a tiny gap (__1.51__) between my GA (`386.1880475`) and the optimal solution (`384.678035`) given by the official team
    - [ ] operators that support to adjuest multiple charging stations in a route are needed
  - [ ] Sequence-based Selection Hyper-heuristic Algorithm
    * Operators needed to revised 
* Reference:
  - (Team 1) Variable Neighbourhood Search by D. Woller, V. Vavra, V. Kozak, M. Kulich
    - [code](https://github.com/wolledav/VNS-EVRP-2020)
    - [papers](http://imr.ciirc.cvut.cz/People/David)
    - latest manuscript, I cannot share here, caz this work of the team hasn't been published
  - (Team 3) Genetic Algorithm by V. Q. Hien, T. C. Dao, T. B. Thang, H. T. T, Binh
    - [code](https://github.com/NeiH4207/EVRP)
    - [paper](https://www.researchgate.net/profile/Cong-Dao-Tran/publication/360604653_A_greedy_search_based_evolutionary_algorithm_for_electric_vehicle_routing_problem/links/641d203a315dfb4ccea54309/A-greedy-search-based-evolutionary-algorithm-for-electric-vehicle-routing-problem.pdf)

## RUN

* an example to run the `GA`

  ```shell
  python run.py -n E-n22-k4.evrp
  ```

* The working directory list

  ```shell
  .
  ├── Figure_1.png
  ├── Figure_2.png
  ├── LICENSE
  ├── Optima_Comparison.png
  ├── README.md
  ├── evrp
  │   ├── __init__.py
  │   ├── evrp_instance.py
  │   ├── ga.py
  │   └── utils.py
  ├── evrp-benchmark-set
  │   ├── E-n101-k8.evrp
  │   ├── E-n22-k4.evrp
  │   ├── E-n23-k3.evrp
  │   ├── E-n30-k3.evrp
  │   ├── E-n33-k4.evrp
  │   ├── E-n51-k5.evrp
  │   ├── E-n76-k7.evrp
  │   ├── X-n1001-k43.evrp
  │   ├── X-n143-k7.evrp
  │   ├── X-n214-k11.evrp
  │   ├── X-n351-k40.evrp
  │   ├── X-n459-k26.evrp
  │   ├── X-n573-k30.evrp
  │   ├── X-n685-k75.evrp
  │   ├── X-n749-k98.evrp
  │   ├── X-n819-k171.evrp
  │   └── X-n916-k207.evrp
  ├── results
  │   ├── E-n22-k4_seed1_nG2.csv
  │   └── E-n22-k4_seed1_nG200.csv
  └── run.py
  ```

  

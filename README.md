# Task II - Generalist Agent

#### Team 1:
Youssef el Ghouch (2568930), Andreea D. Hazu (2645225), Paul Hosek (2753446), Nathan M. Jones (2762057) & Licia Tauriello (2743892)

## Program Overview
The goal of this project is to create a generalist agent who can defeat different adversaries in "Evoman" game framework.
Two evolutionary methods,CMA-ES and MO-CMA-ES, are evaluated. Both methods use DEAP framework to generate the weights of a controller for the 'EvoMan' framework, which consists of a single hidden layer with 10 neurons. Model's hyperparameters like population size and generational number have been tuned using Optuna framework.

## Requirements
* Python 3.10.1+
* numpy
* pandas
* optuna
* math
* deap

### Running code

```sh
python task2.py
```
```sh
python utils.py
```
```sh
python task2_tuning.py
```
```sh
python parameter_tests.py
```
```sh
python bar_enemies.py
```



### Results


```sh
exp_name
│
├── best_results
│   └── best_individuals
├── evoman_logs.txt
└── plots
```

## Key Source Files
> **task2.py**  
> CMA-ES and MO-CMA-ES used to defeat multiple enemies

> **utils.py**  
> defining the fuctions used for tuning of the hyperparameters

> **task2_tuning.py**  
> tuning the hyperparameters

> **parameter_tests.py**  
> testing the parameters from the framework

> **plot_bar_enemies.py**  
> plotting the box-graph with fitness



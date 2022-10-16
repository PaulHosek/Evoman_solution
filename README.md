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
* multiprocessing

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

> **statistical_tests.py**  
> Run pairwise t-test on all enemies for multiple set of weights.
> Run wilcoxon signed-rank test on gain for all enemies.


> **plot_bar_enemies.py**  
> Comparing multiple algorithms over
> multiple runs for all enemies.
> Draws boxplot comparison plot.

> **find_best.py**  
> Attempts to find best set of weights maximising gain, fitness and the 
> number of defeated enemies respectively. Generates and writes three files with best found weights and their values.

> **line_plot.py**  
> Plots the mean and max generational fitness for two EAs

> **box_plot.py**  
> Plots box plot summary of the Gain values for two EAs

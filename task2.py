# --------------------- Import Frameworks and Libraries ---------------------- #
import math
import operator
import sys, os
import json
import utils

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import multiprocessing as mp
from deap import tools, creator, base, algorithms
from os import environ
from deap import cma


print("\nUsing evoman_fast!!! ...vrooom\n")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evoman_fast'))

from evoman.environment import Environment
from demo_controller import player_controller

# Create experiment folder if needed
experiment_name = 'task2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    os.makedirs(experiment_name + '/best_results')
    os.makedirs(experiment_name + '/best_results' + '/best_weights')
    os.makedirs(experiment_name + '/best_results' + '/best_individuals')
    os.makedirs(experiment_name + '/plots')

# Prevent graphics and audio rendering to speed up simulations
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

# Read environment variables
NRUN = int(environ.get("NRUN", 2))
enemies = list(map(int, environ.get("enemy", '7-8').split('-')))
MU = int(environ.get("mu", 10))
LAMBDA = int(environ.get("lambda", 20))
NGEN = int(environ.get("ngen", 500))
multiple_mode = 'yes' if len(enemies) > 1 else 'no'
n_hidden_neurons = 10

# Initialise the game environment for the chosen settings
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  multiplemode=multiple_mode,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# -------------------------------- Functions For DEAP ------------------------- #
# a game simulation for environment env and game x
def simulation(x):
    results = env.play(pcont=x)
    return results

# evaluation of game
def cust_evaluate(x):
    fitness, player_life, enemy_life, game_time = simulation(x)
    return (fitness,)

# Determine individuals that need to be evaluated
def evaluate_pop(pop):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [indiv for indiv in pop if not indiv.fitness.valid]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
    for individual, fit in zip(pop, fitness):
        individual.fitness.values = fit
    return pop

# Needed for HOF
def eq_(var1, var2):
    return operator.eq(var1, var2).all()

# ----------------------------- Initialise DEAP ------------------------------ #

n_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
cma_es = cma.Strategy(centroid=np.random.uniform(-1, 1, n_weights), sigma=math.sqrt(1/LAMBDA), lambda_=LAMBDA)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create('Individual', np.ndarray, fitness=creator.FitnessMax, player_life=100, enemy_life=100)

toolbox = base.Toolbox()
toolbox.register("evaluate", cust_evaluate)
toolbox.register("generate", cma_es.generate, creator.Individual)
toolbox.register("update", cma_es.update)

# Register statistics functions
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("mean", np.mean)
stats.register("max", np.max)

# ---------------------------------- Main ------------------------------------ #
def main():
    # We run the experiment a few times - n_runs
    for run in range(1, NRUN + 1):
        print(f"Start of evolution enemies {enemies} run {run}")

        hof = tools.ParetoFront(eq_)
        pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof)

        print("-- End of (successful) evolution --")

if __name__ == "__main__":
    # Use multiprocessing map implementation for parallelization
    pool = mp.Pool(processes=mp.cpu_count())
    toolbox.register('map', pool.map)

    main()

    pool.close()
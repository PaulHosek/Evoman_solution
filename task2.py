# --------------------- Import Frameworks and Libraries ---------------------- #
import math
import operator
import sys, os
import utils
import numpy as np
import pandas as pd
import multiprocessing as mp

from deap import tools, creator, base, algorithms
from os import environ
from deap import cma

sys.path.insert(0, 'evoman_fast')
from environment import Environment
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
NRUN = int(environ.get("NRUN", 1))
enemies = list(map(int, environ.get("enemy", '7-8').split('-')))
MU = int(environ.get("mu", 10))
LAMBDA = int(environ.get("lambda", 20))
NGEN = int(environ.get("ngen", 500))
multiple_mode = 'yes' if len(enemies) > 1 else 'no'
n_hidden_neurons = 10
strategy = environ.get("strategy", 'cma')

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
# equality test: equivalent to var1 == var2
def eq_(var1, var2):
    return operator.eq(var1, var2).all()

# ----------------------------- Initialise DEAP ------------------------------ #

n_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# all return best parameters for their algorithm; centroid= location of start of evo;
# sigma= initial st.dev of pop; lambda_ = n children produced;
if strategy == 'cma-pl':
    # TO-DO https://deap.readthedocs.io/en/master/api/algo.html#deap.cma.StrategyOnePlusLambda
    cma_es = cma.Strategy(centroid=np.random.uniform(-1, 1, n_weights), sigma=math.sqrt(1/LAMBDA), lambda_=LAMBDA)
# CMA + multi-objective selection method
elif strategy == 'cma-mo':
    # TO-DO https://deap.readthedocs.io/en/master/api/algo.html#deap.cma.StrategyMultiObjective
    cma_es = cma.Strategy(centroid=np.random.uniform(-1, 1, n_weights), sigma=math.sqrt(1 / LAMBDA), lambda_=LAMBDA)
else:
    cma_es = cma.Strategy(centroid=np.random.uniform(-1, 1, n_weights), sigma=math.sqrt(1 / LAMBDA), lambda_=LAMBDA)

if not hasattr(creator, 'FitnessMax'):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
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
    best_individuals = []
    statistics = []

    pool = mp.Pool(processes=mp.cpu_count())
    toolbox.register("map", pool.map)
    # We run the experiment a few times - n_runs
    for run in range(1, NRUN + 1):
        print(f"{strategy} strategy -- Start of evolution enemies {enemies} run {run}")

        # initialise hall of fame
        hof = tools.ParetoFront(eq_)
        # generate new population + update its value with methods in toolbox
        # logbook = statistics of the evolution
        pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof)

        best_individuals.append(hof[0])
        statistics.append(pd.DataFrame(logbook))

        print("-- End of (successful) evolution --")

        # Write best individuals fitness values for enemy and experiment
        exp_name = f"{MU}_{LAMBDA}_{NGEN}"
        utils.plot_exp_stats(statistics, experiment_name + "/plots/", exp_name, strategy, str(enemies), NGEN)
        utils.write_best(best_individuals, experiment_name + "/best_results/best_weights/", exp_name, strategy, str(enemies))
        utils.eval_best(best_individuals, experiment_name + "/best_results/best_individuals/", exp_name, strategy, str(enemies))
    pool.close()

if __name__ == "__main__":
    main()
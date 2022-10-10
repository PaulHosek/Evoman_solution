# Hyperparameters to Tune

# - SIGMA = Initial step size (float)
# - MU = The number of parents to keep from the lambda children (integer)
# - LAMBDA = Number of children to produce at each generation (integer)

# --------------------- Import Frameworks and Libraries ---------------------- #
import math
import operator
import sys, os
import utils
import numpy as np
import pandas as pd
import multiprocessing as mp
import optuna

from deap import tools, creator, base, algorithms
from os import environ
from deap import cma

sys.path.insert(0, 'evoman_fast') # TODO - what does evoman_fast do?

from environment import Environment
from demo_controller import player_controller

# ---------------------------------- Setup ---------------------------------- #

# Create required folders
experiment_name = 'task2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

if not os.path.exists(experiment_name + "/tuning_results"):
    os.makedirs(experiment_name + "/tuning_results")

# Prevent graphics and audio rendering to speed up simulations
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

# Read environment variables
NTRIALS = int(environ.get("NTRIALS", 10))
enemies = list(map(int, environ.get("enemy", '6-7-8').split('-')))
NGEN = int(environ.get("ngen", 5))
multiple_mode = 'yes' if len(enemies) > 1 else 'no'
n_hidden_neurons = 10
strategy = environ.get("strategy", "cma-mo") # ["cma-mo", "cma-opl", "cma"] 

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
def cust_evaluate(indiv):
    fitness, player_life, enemy_life, game_time = simulation(indiv)
    return (fitness,)

# evaluation of an Individual weights when playing against enemies
# we need all fitness values to send for the MO strategy
def cust_evaluate_mo(indiv):
    mo_fitness = []
    for idx, enemy in enumerate(enemies):
        fit, _, _, _ = env.run_single(enemy, pcont=np.array(indiv), econt="None")
        mo_fitness.append(fit)
    # we return a tuple of size len(enemies) with the fitness values after each game
    return tuple(mo_fitness)

# Needed for HOF
def eq_(var1, var2):
    return operator.eq(var1, var2).all()

# ----------------------------- Initialise DEAP ------------------------------ #

n_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
# print("Num Weights:", n_weights)

if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "Individual"):
    del creator.Individual

toolbox = base.Toolbox()
if strategy == 'cma-mo':
    # create the fitness attribute of an Individual as a tuple of size len(enemies)
    # we will not have one fitness value as before to maximize, but len(enemies) fitness values
    # these are our objectives - the fitness values for playing against each enemy from the enemies list
    # i.e.: for 3 enemies, weights(or fitness) will be a tuple(1.0,1.0,1.0)
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,) * len(enemies))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti, player_life=100, enemy_life=100)
    toolbox.register("evaluate", cust_evaluate_mo)
else:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMax, player_life=100, enemy_life=100)
    toolbox.register("evaluate", cust_evaluate)

# # TO-DO: add seed for reproducibility
# if strategy == 'cma-mo':
#     # Generate population of MU individuals from an Uniform distribution -1, 1 with n_weights
#     population = [creator.Individual(x) for x in (np.random.uniform(-1, 1, (MU, n_weights)))]
#     # We evaluate the initial population and update its fitness
#     fitnesses = toolbox.map(toolbox.evaluate, population)
#     for ind, fit in zip(population, fitnesses):
#         ind.fitness.values = fit
#     # We initialize the MO strategy
#     cma_strategy = cma.StrategyMultiObjective(population, sigma=math.sqrt(1 / LAMBDA), mu=MU, lambda_=LAMBDA)
# elif strategy == 'cma-opl':
#     parent = creator.Individual(np.random.uniform(-1, 1, n_weights))
#     parent.fitness.values = toolbox.evaluate(parent)
#     cma_strategy = cma.StrategyOnePlusLambda(parent, sigma=math.sqrt(1 / LAMBDA), lambda_=LAMBDA)
# else:
#     cma_strategy = cma.Strategy(centroid=np.random.uniform(-1, 1, n_weights), sigma=math.sqrt(1 / LAMBDA), lambda_=LAMBDA)

# toolbox.register("generate", cma_strategy.generate, creator.Individual)
# toolbox.register("update", cma_strategy.update)

# Register statistics functions
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("mean", np.mean)
stats.register("max", np.max)

# ---------------------------- Tuning Objective ----------------------------- #

def get_strategy(strategy, trial):
    # TO-DO: add seed for reproducibility
    if strategy == 'cma-mo':
        # Tunable parameters
        sigma = trial.suggest_float('SIGMA', 1e-3, 0.5e1, log=True)
        mu = trial.suggest_int('MU', 5, 30)
        lam = trial.suggest_int('LAMBDA', 2 * mu, 7 * mu)

        # Generate population of MU individuals from an Uniform distribution -1, 1 with n_weights
        population = [creator.Individual(x) for x in (np.random.uniform(-1, 1, (mu, n_weights)))]
        # We evaluate the initial population and update its fitness
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # We initialize the MO strategy
        cma_strategy = cma.StrategyMultiObjective(population, sigma=sigma, mu=mu, lambda_=lam)
        return cma_strategy
    elif strategy == 'cma-opl':
        # Tunable parameters
        sigma = trial.suggest_float('SIGMA', 1e-3, 0.5e1, log=True)
        lam = trial.suggest_int('LAMBDA', 1, 5)
        
        parent = creator.Individual(np.random.uniform(-1, 1, n_weights))
        parent.fitness.values = toolbox.evaluate(parent)

        cma_strategy = cma.StrategyOnePlusLambda(parent, sigma=sigma, lambda_=lam)
        return cma_strategy
    else:
        # Tunable parameters
        sigma = trial.suggest_float('SIGMA', 1e-3, 0.5e1, log=True)
        mu = trial.suggest_int('MU', 5, 30)
        lam = trial.suggest_int('LAMBDA', 2 * mu, 7 * mu)

        cma_strategy = cma.Strategy(centroid=np.random.uniform(-1, 1, n_weights), sigma=sigma, mu=mu, lambda_=lam)
        return cma_strategy


def objective(trial):
    # Define strategy
    cma_strategy = get_strategy(strategy, trial)
    toolbox.register("generate", cma_strategy.generate, creator.Individual)
    toolbox.register("update", cma_strategy.update)

    hof = tools.ParetoFront(eq_)

    # generate new population + update its value with methods in toolbox
    # logbook = statistics of the evolution

    # Expanded version of algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof)
    # ----------------------------------------------------------------------- #
    # Function inputs (toolbox and stats are global objects with the same names):
    ngen = NGEN
    halloffame = hof

    # Function implementation:
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        print(logbook.stream) # Comment out to hide generation tracking

        # return population, logbook
        
        # Inserted logic for optuna pruning (early stopping for poor trials)
        generation, mean = logbook.select("gen", "mean")
        trial.report(mean[-1], generation[-1])

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # ----------------------------------------------------------------------- #

    return mean[-1]


# ---------------------------------- Main ----------------------------------- #
def main():
    exp_name = f"strat-{strategy}_enemies-{enemies}_gen-{NGEN}_trials-{NTRIALS}"
    study = optuna.create_study(study_name=exp_name, direction="maximize")
    study.optimize(objective, n_trials=NTRIALS)
    utils.print_tuning_results(study)
    utils.save_tuning_results(study, f"{experiment_name}/tuning_results/{exp_name}.txt", strategy, enemies, NGEN)

if __name__ == "__main__":
    pool = mp.Pool(processes=mp.cpu_count())
    toolbox.register("map", pool.map)
    main()
    pool.close()
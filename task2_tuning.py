# ------------------------------ Dependencies ------------------------------- #
# 
# Some extra python modules are required for optuna's visualization tools.
# Run the following commands to install them:
# 
# > pip install plotly==5.10.0
# > pip install -U scikit-learn
# > pip install -U kaleido
# 
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

sys.path.insert(0, 'evoman_fast') # TODO - what does evoman_fast do? How?

from environment import Environment
from demo_controller import player_controller

# ---------------------------------- Setup ---------------------------------- #

# Read environment variables
STRATEGY = environ.get("STRATEGY", "cma-mo") # ["cma-mo", "cma-opl", "cma"] 
ENEMIES = list(map(int, environ.get("enemy", '1-2-3-4-7').split('-'))) # ["1-2-3-4-5-6-7-8"]
NGEN = int(environ.get("NGEN", 50))
NTRIALS = int(environ.get("NTRIALS", 100))
WINDOW_LEN = 5

MULTIPLE_MODE = 'yes' if len(ENEMIES) > 1 else 'no'
N_HIDDEN_NEURONS = 10

# Hyperparameter search space
# SIGMA = Initial step size (float)
SIGMA_LOWER = 1e-3
SIGMA_UPPER = 2
SIGMA_LOG = True

# - MU = The number of parents to keep from the lambda children (integer)
MU_LOWER = 5
MU_UPPER = 30

# ['cma-opl' ONLY] LAMBDA = Number of children to produce at each generation (integer)
LAMBDA_LOWER = 1
LAMBDA_UPPER = 30

# Store settings in dict for recording purposes
exp_settings = {
    "Strategy": STRATEGY,
    "Enemies": ENEMIES,
    "Generations": NGEN,
    "Trials": NTRIALS,
    "Window Length": WINDOW_LEN,
    "Sigma_lower": SIGMA_LOWER,
    "Sigma_upper": SIGMA_UPPER,
    "Sigma_log": SIGMA_LOG,
    "Mu_lower": MU_LOWER,
    "Mu_upper": MU_UPPER,
    "Lambda_lower": LAMBDA_LOWER,
    "Lambda_upper": LAMBDA_UPPER
}

# Create required folders
root_folder = 'task2'
exp_name = f"strat-{STRATEGY}_enemies-{ENEMIES}_gen-{NGEN}_trials-{NTRIALS}"
tuning_subfolder = f"{root_folder}/tuning_results/{exp_name}"

if not os.path.exists(root_folder):
    os.makedirs(root_folder)

if not os.path.exists(f"{root_folder}/tuning_results"):
    os.makedirs(f"{root_folder}/tuning_results")

if not os.path.exists(tuning_subfolder):
    os.makedirs(tuning_subfolder)

# Prevent graphics and audio rendering to speed up simulations
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

# Initialise the game environment for the chosen settings
env = Environment(experiment_name=root_folder,
                  enemies=ENEMIES,
                  multiplemode=MULTIPLE_MODE,
                  playermode="ai",
                  player_controller=player_controller(N_HIDDEN_NEURONS),
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
    for idx, enemy in enumerate(ENEMIES):
        fit, _, _, _ = env.run_single(enemy, pcont=np.array(indiv), econt="None")
        mo_fitness.append(fit)
    # we return a tuple of size len(ENEMIES) with the fitness values after each game
    return tuple(mo_fitness)

# Needed for HOF
def eq_(var1, var2):
    return operator.eq(var1, var2).all()

def mean_mo(fit):
    enemy_mean = np.mean(fit, axis=1)
    enemy_std = np.std(fit, axis=1)
    cons_fit = np.subtract(enemy_mean, enemy_std)
    mean_cons_fit = np.mean(cons_fit)
    return mean_cons_fit

def max_mo(fit):
    enemy_mean = np.mean(fit, axis=1)
    enemy_std = np.std(fit, axis=1)
    cons_fit = np.subtract(enemy_mean, enemy_std)
    max_cons_fit = np.max(cons_fit)
    return max_cons_fit

# ----------------------------- Initialise DEAP ------------------------------ #

n_weights = (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5

if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "Individual"):
    del creator.Individual

toolbox = base.Toolbox()
if STRATEGY == 'cma-mo':
    # create the fitness attribute of an Individual as a tuple of size len(ENEMIES)
    # we will not have one fitness value as before to maximize, but len(ENEMIES) fitness values
    # these are our objectives - the fitness values for playing against each enemy from the enemies list
    # i.e.: for 3 enemies, weights(or fitness) will be a tuple(1.0,1.0,1.0)
    creator.create("FitnessMulti", base.Fitness, weights=(1.0,) * len(ENEMIES))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti, player_life=100, enemy_life=100)
    toolbox.register("evaluate", cust_evaluate_mo)
else:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMax, player_life=100, enemy_life=100)
    toolbox.register("evaluate", cust_evaluate)

# Register statistics functions
stats = tools.Statistics(lambda ind: ind.fitness.values)

if STRATEGY == 'cma-mo':
    stats.register("mean", mean_mo)
    stats.register("max", max_mo)
else:
    stats.register("mean", np.mean)
    stats.register("max", np.max)

# -------------------------- Functions for optuna --------------------------- #

# Returns an initialized DEAP object for the chosen strategy
def get_strategy(strategy, trial):
    # TO-DO: add seed for reproducibility
    if strategy == 'cma-mo':
        # Tunable parameters
        sigma = trial.suggest_float('SIGMA', SIGMA_LOWER, SIGMA_UPPER, log=SIGMA_LOG)
        mu = trial.suggest_int('MU', MU_LOWER, MU_UPPER)
        lam = trial.suggest_int('LAMBDA', 1, 5 * mu)

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
        sigma = trial.suggest_float('SIGMA', SIGMA_LOWER, SIGMA_UPPER, log=SIGMA_LOG)
        lam = trial.suggest_int('LAMBDA', LAMBDA_LOWER, LAMBDA_UPPER)
        
        parent = creator.Individual(np.random.uniform(-1, 1, n_weights))
        parent.fitness.values = toolbox.evaluate(parent)

        cma_strategy = cma.StrategyOnePlusLambda(parent, sigma=sigma, lambda_=lam)
        return cma_strategy

    else:
        # Tunable parameters
        sigma = trial.suggest_float('SIGMA', SIGMA_LOWER, SIGMA_UPPER, log=SIGMA_LOG)
        mu = trial.suggest_int('MU', MU_LOWER, MU_UPPER)
        lam = trial.suggest_int('LAMBDA', mu, 5 * mu)

        cma_strategy = cma.Strategy(centroid=np.random.uniform(-1, 1, n_weights), sigma=sigma, mu=mu, lambda_=lam)
        return cma_strategy

# The function for optuna to maximise (mean fitness after NGEN generations)
# Implements pruning/early stopping for poor trials
def objective(trial):
    # Define strategy
    cma_strategy = get_strategy(STRATEGY, trial)
    toolbox.register("generate", cma_strategy.generate, creator.Individual)
    toolbox.register("update", cma_strategy.update)

    hof = tools.ParetoFront(eq_)

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
        smoothed_mean = sum(mean[-WINDOW_LEN:]) / len(mean[-WINDOW_LEN:])
        trial.report(smoothed_mean, generation[-1])

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # ----------------------------------------------------------------------- #

    return smoothed_mean

# ---------------------------------- Main ----------------------------------- #
def main():
    # Tuning
    study = optuna.create_study(study_name=exp_name, direction="maximize")
    study.optimize(objective, n_trials=NTRIALS)

    # Saving and displaying results
    utils.save_study(study, tuning_subfolder)
    utils.print_tuning_summary(study, exp_settings)
    utils.save_tuning_summary(study, tuning_subfolder, exp_settings)
    utils.save_tuning_plots(study, tuning_subfolder)
    

if __name__ == "__main__":
    pool = mp.Pool(processes=mp.cpu_count())
    toolbox.register("map", pool.map)
    main()
    pool.close()
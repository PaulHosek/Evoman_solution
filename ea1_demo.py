# imports framework
import itertools
import sys, os
import numpy as np
import json
import time
import matplotlib.pyplot as plt


if os.environ.get('EVOMAN_FAST'):
    print("\nUsing evoman_fast!!! ...vrooom\n")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evoman_fast'))
else:
    print("\nUsing standard evoman\n")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evoman'))

from evoman.environment import Environment
from demo_controller import player_controller
from deap import tools, creator, base, algorithms
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# create experiment folder if needed
experiment_name = 'ea_exp'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    os.makedirs(experiment_name + '/best_results')
    os.makedirs(experiment_name + '/plots')

# DEFINE VARIABLES
# environment variables
enemies = [1] #list of player enemies - any from 1..8
n_runs = 10 #should be 10
gen_size = 30 #should be 100
pop_size = 100
n_hidden_neurons = 10
max_budget = 500 #default is 3000
difficulty_level = 2 #default is 1
enemymode = "static" #default is ai
players_life = 100
#deap_algorithms = ['eaMuPlusLambda', 'eaMuCommaLambda', 'eaSimple']
deap_algorithms = ['eaMuCommaLambda']

# deap variables
mating_prob = 1
crossover_rate = 0.8
mutation = 0.2
toolbox, log = None, None

# returns environment for specific enemies
def get_env(enemies):
    # initializes environments for each game/enemy type with ai player using random controller, playing against static enemy
    return Environment(experiment_name=experiment_name,
                     enemies=enemies,
                     player_controller=player_controller(n_hidden_neurons),
                     enemymode="static",
                     level=difficulty_level,
                     speed="fastest",
                     timeexpire=max_budget)

# a game simulation for environment env and game x
def simulation(env,x):
    fitness, player_life, enemy_life, game_time = env.play(pcont=x)
    return fitness

# evaluation of game
def cust_evaluate(env,x):
    sim = simulation(env,x)
    return (sim,)

# Determine individuals that need to be evaluated
def evaluate_pop(env, pop):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [indiv for indiv in pop if not indiv.fitness.valid]
    # pop_size Environment objects
    envs = [env for i in range(len(invalid_ind))]
    fitness = toolbox.map(toolbox.evaluate, envs, invalid_ind)
    for indiv, fit in zip(pop, fitness):
        indiv.fitness.values = fit
    return pop

# get statistics about game
def get_stats(pop, gen):
    fitness_vals = [individual.fitness.values[0] for individual in pop]
    mean_fit = sum(fitness_vals) / len(pop)
    max_fit = np.max(fitness_vals)
    print("---- Gen: %i --> Mean: %.2f ,Max %.2f" % (gen, mean_fit, max_fit))
    return mean_fit, max_fit

# write statistics
def write_stats_in_file(stats, file):
    f = open(file, "w+")
    for stat in stats:
        f.write(json.dumps(stat) + "\n")

# Record statistics in deap logger
def record_stat(pop, generation, run, enemy, best):
    global log
    best.update(pop)
    mean_fit, max_fit = get_stats(pop, generation)
    statistic = {"mean": mean_fit, "max": max_fit}
    log.record(enemy=enemy, run=run, gen=generation, individuals=len(pop), **statistic)
    return [mean_fit, max_fit]

# Write best individuals in file
def write_best(individuals, file, enemy):
    for count, individual in enumerate(individuals):
        tmp = count + 1
        f = open(file + "_e" + str(enemy) + "_i" + str(tmp) + ".txt", "w+")
        for weight in individual:
            f.write(str(weight) + "\n")

def init_deap():
    global toolbox, log

    # deap - creating types: Fitness, Individual and Population
    # Fitness - tuple, we give one for single objective, 1.0 for maximizing
    creator.create("FitnessBest", base.Fitness, weights=(1.0,))
    # Individual class inherited from numpy.ndarray
    creator.create('Individual', np.ndarray, fitness=creator.FitnessBest, player_life=players_life, enemy_life=players_life)

    # OPERATORS & INITIALIZATIONS OF CLASSES
    toolbox = base.Toolbox()
    # population drawn from uniform distribution
    toolbox.register("indices", np.random.uniform, -1, 1)

    # Register functions
    toolbox.register("evaluate", cust_evaluate)
    toolbox.register("mate", tools.cxTwoPoint) # crossover operator
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
    toolbox.register("select", tools.selTournament,tournsize=2)

    # Initialize deap logbook
    log = tools.Logbook()
    log.header = ['enemy','run','generation','population size','fitness mean','fitness max']

def plot_exp_stats(enemy, statistics, alg):
    # TO-DO:
    # Save statistics before plotting in file so we can play more with the experiments data
    x = range(1, gen_size + 1)
    means = np.transpose(np.mean(statistics, axis=0)) #this migjt be automatically done with seaborn
    # stds = np.transpose(np.std(statistics, axis=0))

    plt.figure(figsize=(10, 8))
    plt.title("%s Enemy %i - Average and Maximum Fitness of each Generation" % (alg, enemy))
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.plot(x, means[0], color="red", label="Mean Fitness")
    plt.plot(x, means[1], color="blue", label="Maximum Fitness")
    plt.legend(loc="lower right")
    plt.savefig(experiment_name + '/plots/' + alg + '_enemy' + str(enemy) + '.png')
    plt.ylim(0, 100)
    plt.show()

# TO-DO: simplify this and call the existing DEAP functions
def create_next_generation(env, pop, alg="eaSimple"):
    global pop_size
    #mu and lambda are pop_size
    if alg == 'eaSimple':
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Vary the pool of individuals
        algorithms.eaSimple()
        offspring = algorithms.varAnd(offspring, toolbox, mating_prob, mutation)
        # Evaluate the individuals with an invalid fitness
        offspring = evaluate_pop(env, offspring)
        # Replace the current population by the offspring
        pop[:] = offspring
    elif alg == 'eaMuPlusLambda':
         # Vary the population
         offspring = algorithms.varOr(pop, toolbox, pop_size, crossover_rate, mutation)
         # Evaluate the individuals with an invalid fitness
         offspring = evaluate_pop(env, offspring)
         # Select the next generation population
         pop[:] = toolbox.select(pop + offspring, pop_size)
    elif alg == 'eaMuCommaLambda':
         # Vary the population
         offspring = algorithms.varOr(pop, toolbox, pop_size, crossover_rate, mutation)
         # Evaluate the individuals with an invalid fitness
         offspring = evaluate_pop(env, offspring)
         # Select the next generation population
         pop[:] = toolbox.select(offspring, pop_size)

    # get best results
    best = tools.HallOfFame(1, similar=np.array_equal)
    return pop, best

def main():
    init_deap()
    for alg in deap_algorithms:
        print("Algorithm: %s" % alg)
        # For each of the n enemies we want to run the experiment for:
        for enemy in enemies:
            # Get enviroment for player and prepare DEAP
            env = get_env([enemy])
            # number of weights for multilayer network with n_hidden_neurons
            n_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.indices, n=n_weights)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)

            best_individuals = []
            statistics = []

            # We run the experiment a few times - n_runs
            for run in range(1,n_runs+1):
                start_time = time.time()
                gen_stat = []

                # ---------------Initialize population ---------------
                #pop = np.random.uniform(low=-1, high=1, size=(n_population, n_weights))
                pop = toolbox.population(n=pop_size)
                #evaluate first generation
                pop = evaluate_pop(env, pop)
                best = tools.HallOfFame(1, similar=np.array_equal)
                # record these best results in file
                stat = record_stat(pop, generation=0, run=run, enemy=enemy, best=best)
                gen_stat.append(stat)

                print("Start of evolution player %i run %i" % (enemy, run))
                for generation in range(1, gen_size):
                    pop, best = create_next_generation(env, pop, alg)
                    # record these best results in file
                    stat = record_stat(pop, generation=generation, run=run, enemy=enemy, best=best)
                    gen_stat.append(stat)

                print("-- End of (successful) evolution --")
                statistics.append(gen_stat)
                best_individuals.append(best[0])
                print("---- %s seconds elapsed ----" % (time.time() - start_time))

            # Write best individuals fitness values for enemy and experiment
            write_best(best_individuals, experiment_name + "/best_results/Best_individuals_" + experiment_name + alg, enemy)
            plot_exp_stats(enemy, statistics, alg)

        # Write statistics for experiment
        write_stats_in_file(log,"log_stats_" + experiment_name + alg + ".txt")

if __name__ == "__main__":
    main()
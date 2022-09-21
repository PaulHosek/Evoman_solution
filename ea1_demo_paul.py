# imports framework
import itertools
import sys, os
import numpy as np
import json
import time
import matplotlib.pyplot as plt

sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller
from deap import tools, creator, base, algorithms
os.environ["SDL_VIDEODRIVER"] = "dummy"


# create experiment folder if needed
experiment_name = 'EA1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    os.makedirs(experiment_name + '/best_results')
    os.makedirs(experiment_name + '/plots')

# DEFINE VARIABLES
# environment variables
enemies = [1, 4, 8] #list of player enemies - any from 1..8
n_runs = 1 #should be 10
gen_size = 2 #should be 100
pop_size = 10
n_hidden_neurons = 10
max_budget = 500 #default is 3000
difficulty_level = 2 #default is 1
enemymode = "static" #default is ai
players_life = 100

# deap variables
mate = 1
mutation = 0.2
toolbox, log = None, None
eta = 2
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
    # pop_size Individual objects with weights
    individuals = [indiv for indiv in pop if not indiv.fitness.valid]
    # pop_size Environment objects
    envs = [env for i in range(len(individuals))]
    print("---- Will evaluate %i individuals" % len(individuals))
    fitnesses = toolbox.map(toolbox.evaluate, envs, individuals)
    for indiv, fitness in zip(pop, fitnesses):
        indiv.fitness.values = fitness

# get statistics about game
def get_stats(pop):
    fitness_vals = [individual.fitness.values[0] for individual in pop]
    mean_fit = sum(fitness_vals) / len(pop)
    max_fit = np.max(fitness_vals)
    print("---- Mean: %.2f ,Max %.2f" % (mean_fit, max_fit))
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
    mean_fit, max_fit = get_stats(pop)
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



# def p_crossover(pop, p_crossover = 0.5, len_pop = pop_size):
#     # 265 weights per individual
#
#     # select
#     print(type(pop[1]))
#     n_samples = np.random.binomial(len_pop,p_crossover)
#     pop_n = np.random.choice(pop, p=p_crossover, replace=True)
#
#     print(pop_n)
#
#     i = 0
#     j = 1
#     raise KeyboardInterrupt
#     # while j < len_pop-1:
#     #     np.random
#     #     i += 2
#     #     j += 2
# "Rather than selecting offspring values uniformly from a range around each parent values,
# they are selected from a distribution which is more likely to create small changes,
# and the distribution is controlled by the distance between the parents."
def sbx(p1, p2, eta=eta):
    """
    Simulated Binary Crossover for 2 individuals

    @param p1, p2: parent 1 of class individual
    @param eta:  non-negative index of distribution; paper used 2 and 5. Higher = more similarity to parents
    """
    # based on "A Taxonomy for the Crossover Operator for Real-Coded Genetic Algorithms: An Experimental Study"
    def gen_beta(eta):
        # uniform number source u(0,1)
        u = np.random.random()
        if u <= 0.5:
            return (2.0 * u) ** (1.0 / (eta + 1.0))
        return 1.0 / (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))

    for idx, (p1_val, p2_val) in enumerate(zip(p1, p2)):
            beta = gen_beta(eta)
            p2[idx] = 0.5 * ((1.0 - beta) * p1_val + (1.0 + beta) * p2_val)
            p1[idx] = 0.5 * ((1.0 + beta) * p1_val + (1.0 - beta) * p2_val)

    return p1, p2





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
    # toolbox.register("mate", tools.cxTwoPoint) # crossover operator
    toolbox.register("mate", tools.cxTwoPoint) # crossover operator
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
    toolbox.register("select", tools.selRoulette)

    # Initialize deap logbook
    log = tools.Logbook()
    log.header = ['enemy','run','generation','population size','fitness mean','fitness max']

def plot_exp_stats(enemy, statistics):
    x = range(1, gen_size + 1)
    means = np.transpose(np.mean(statistics, axis=0))
    # stds = np.transpose(np.std(statistics, axis=0))

    plt.title("%s Enemy %i - Average and Maximum Fitness of each Generation" % (experiment_name, enemy))
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.plot(x, means[0], color="red", label="Mean Fitness")
    plt.plot(x, means[1], color="blue", label="Maximum Fitness")
    plt.legend(loc="lower right")
    plt.savefig(experiment_name + '/plots/enemy' + str(enemy) + '.png')
    plt.show()

def main():
    init_deap()
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
            print("Start of evolution player %i run %i" % (enemy, run))
            for generation in range(1, gen_size+1):
                # ------- Evaluate current generation -------- #
                print("-- Generation %i --" % generation)
                # p_crossover(pop)
                # evaluate fitness for population
                evaluate_pop(env, pop)

                # get best results
                best = tools.HallOfFame(1, similar=np.array_equal)
                # record these best results in file
                stat = record_stat(pop,generation=generation,run=run,enemy=enemy,best=best)
                gen_stat.append(stat)


                # ---------------Create the next generation by crossover and mutation --------------- #
                if generation < gen_size:  # not necessary for the last generation
                    # copy and select individuals to generate offspring
                    offs = map(toolbox.clone, toolbox.select(pop,len(pop))) # select only needed if select subset
                    offs = algorithms.varAnd(pop,toolbox,mate,mutation)
                    pop = offs

            print("-- End of (successful) evolution --")
            statistics.append(gen_stat)
            best_individuals.append(best[0])
            print("---- %s seconds elapsed ----" % (time.time() - start_time))

        # Write best individuals fitness values for enemy and experiment
        write_best(best_individuals, experiment_name + "/best_results/Best_individuals_" + experiment_name, enemy)
        plot_exp_stats(enemy, statistics)

    # Write statistics for experiment
    write_stats_in_file(log,"log_stats_" + experiment_name + ".txt")

if __name__ == "__main__":
    main()
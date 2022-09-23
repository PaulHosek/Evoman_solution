# --------------------- Import Frameworks and Libraries ---------------------- #
import operator
import sys, os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from deap import tools, creator, base, algorithms

if os.environ.get('EVOMAN_FAST'):
    print("\nUsing evoman_fast!!! ...vrooom\n")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evoman_fast'))
else:
    print("\nUsing standard evoman\n")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evoman'))

from evoman.environment import Environment
from demo_controller import player_controller

# ---------------------------------- Setup ----------------------------------- #

# Prevent graphics and audio rendering to speed up simulations
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Create experiment folder if needed
experiment_name = 'ea_exp'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    os.makedirs(experiment_name + '/best_results')
    os.makedirs(experiment_name + '/plotted_results')
    os.makedirs(experiment_name + '/plots')

# DEFINE VARIABLES
# NRUN = 10 #should be 10
# NGEN = 30 #should be 100?
# MU = 100
# LAMBDA = 100
# CXPB = 0.8
# MUTPB = 0.2
NRUN = 2 #should be 10
NGEN = 40 #should be 100?
MU = 50
LAMBDA = 50
CXPB = 0.8
MUTPB = 0.2

enemies = [1] #list of player enemies - any from 1-8
n_hidden_neurons = 10
max_budget = 500 #default is 3000
difficulty_level = 2 #default is 1
enemymode = "static" #default is ai
players_life = 100

#deap_algorithms = ['eaMuPlusLambda', 'eaMuCommaLambda', 'eaSimple']
deap_algorithms = ['eaMuPlusLambda']

# Initialise the game environment for the chosen settings
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=difficulty_level,
                  speed="fastest",
                  timeexpire=max_budget)

# -------------------------------- Functions --------------------------------- #

# a game simulation for environment env and game x
def simulation(env,x):
    fitness, player_life, enemy_life, game_time = env.play(pcont=x)
    return fitness

# evaluation of game
def cust_evaluate(x):
    fitness = simulation(env,x)
    return (fitness,)

# Determine individuals that need to be evaluated
def evaluate_pop(pop):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [indiv for indiv in pop if not indiv.fitness.valid]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
    for individual, fit in zip(pop, fitness):
        individual.fitness.values = fit
    return pop

# write statistics
def write_stats_in_file(stats, file):
    f = open(file, "w+")
    for stat in stats:
        f.write(json.dumps(stat) + "\n")

# Write best individuals in file
def write_best(individuals, file, enemy):
    for count, individual in enumerate(individuals):
        tmp = count + 1
        f = open(file + "_e" + str(enemy) + "_i" + str(tmp) + ".txt", "w+")
        for weight in individual:
            f.write(str(weight) + "\n")

def plot_exp_stats(enemy, statistics, alg):
    # TO-DO:
    # Save statistics before plotting in file so we can play more with the experiments data
    df_stat = pd.concat(statistics)

    # Avg and std of means
    avg_mean = df_stat.groupby(['gen'])['mean'].mean().to_numpy()
    std_mean = df_stat.groupby(['gen'])['mean'].std().to_numpy()
    avg_mean_plus_std = [a + b for a, b in zip(avg_mean, std_mean)]
    avg_mean_minus_std = [a - b for a, b in zip(avg_mean, std_mean)]

    # Avg and std of maxes
    avg_max = df_stat.groupby(['gen'])['max'].mean().to_numpy()
    std_max = df_stat.groupby(['gen'])['max'].std().to_numpy()
    avg_max_plus_std = [a + b for a, b in zip(avg_max, std_max)]
    avg_max_minus_std = [a - b for a, b in zip(avg_max, std_max)]

    gen = range(0, NGEN + 1)
    print(gen)

    # Generate line plot
    fig, ax = plt.subplots()
    ax.plot(gen, avg_mean, '-', label='average mean')
    ax.fill_between(gen, avg_mean_minus_std, avg_mean_plus_std, alpha=0.2)
    ax.plot(gen, avg_max, '-', label='average max')
    ax.fill_between(gen, avg_max_minus_std, avg_max_plus_std, alpha=0.2)
    ax.legend(loc='lower right', prop={'size': 15})
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    ax.grid()
    plt.show()
    
    # print(df_stat.mean())
    exit('Need to play with dataframes for plotting')

    x = range(1, NGEN + 1)
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

# ----------------------------- Initialise DEAP ------------------------------ #

# deap - creating types: Fitness, Individual and Population
# Fitness - tuple, we give one for single objective, 1.0 for maximizing
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Individual class inherited from numpy.ndarray
creator.create('Individual', np.ndarray, fitness=creator.FitnessMax, player_life=players_life, enemy_life=players_life)

# OPERATORS & INITIALIZATIONS OF CLASSES
toolbox = base.Toolbox()
# population drawn from uniform distribution
toolbox.register("indices", np.random.uniform, -1, 1)

# number of weights for multilayer network with n_hidden_neurons
n_weights = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.indices, n=n_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=MU)

# Register EA functions
toolbox.register("evaluate", cust_evaluate)
toolbox.register("mate", tools.cxTwoPoint) # crossover operator
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
toolbox.register("select", tools.selTournament,tournsize=2)

# Register statistics functions
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("mean", np.mean)
stats.register("max", np.max)

# ---------------------------------- Main ------------------------------------ #

def eq_(var1, var2):
    return operator.eq(var1, var2).all()

def main():
    for alg in deap_algorithms:
        print("Algorithm: %s" % alg)
        # For each of the n enemies we want to run the experiment for:
        for enemy in enemies:
            
            best_individuals = []
            statistics = []

            # We run the experiment a few times - n_runs
            for run in range(1, NRUN + 1):
                # ---------------Initialize population ---------------
                #pop = np.random.uniform(low=-1, high=1, size=(n_population, n_weights))
                pop = toolbox.population(n=MU)
                hof = tools.ParetoFront(eq_)

                print("Start of evolution player %i run %i" % (enemy, run))
                if alg == 'eaMuPlusLambda':
                    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                                         cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                                         stats=stats, halloffame=hof)
                elif alg == 'eaMuCommaLambda':
                    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                                         cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                                         stats=stats, halloffame=hof)
                elif alg == 'eaSimple':
                    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                                        stats=stats, halloffame=hof)

                print("-- End of (successful) evolution --")
                best_individuals.append(hof[0])
                statistics.append(pd.DataFrame(logbook))

            print(statistics)
            # Write best individuals fitness values for enemy and experiment
            write_best(best_individuals, experiment_name + "/best_results/Best_individuals_" + experiment_name + alg, enemy)
            plot_exp_stats(enemy, statistics, alg)

        # Write statistics for experiment
        #write_stats_in_file(log,"log_stats_" + experiment_name + alg + ".txt")

if __name__ == "__main__":
    pool = mp.Pool(processes=mp.cpu_count())
    toolbox.register('map', pool.map)

    main()

    pool.close()
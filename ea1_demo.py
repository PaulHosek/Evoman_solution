# --------------------- Import Frameworks and Libraries ---------------------- #
import operator
import sys, os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import multiprocessing as mp
from deap import tools, creator, base, algorithms
from os import environ


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
environ["SDL_VIDEODRIVER"] = "dummy"
environ["SDL_AUDIODRIVER"] = "dummy"
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Create experiment folder if needed
experiment_name = 'ea_exp'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    os.makedirs(experiment_name + '/best_results')
    os.makedirs(experiment_name + '/best_results' + '/best_weights')
    os.makedirs(experiment_name + '/best_results' + '/best_individuals')
    os.makedirs(experiment_name + '/plots')

# DEFINE VARIABLES
NRUN = 10 #should be 10

enemies = [1] #list of player enemies - any from 1-8
n_hidden_neurons = 10
max_budget = 500 #default is 3000
enemymode = "static" #default is ai
players_life = 100

# Read environment variables
selection = environ.get("sel", 'selTournament')
mutation = environ.get("mut", 'mutShuffleIndexes')
crossover = environ.get("cx", 'cxTwoPoint')
MU = int(environ.get("mu", 100))
LAMBDA = int(environ.get("lambda", 100))
NGEN = int(environ.get("ngen", 30))
CXPB = round(float(environ.get("cxpb", 0.8)),2)
MUTPB = round(1-CXPB, 2)

# Manual override
NRUN = 3
NGEN = 5
MU = 10
LAMBDA = 10

#deap_algorithms = ['eaMuPlusLambda', 'eaMuCommaLambda', 'eaSimple']
deap_algorithms = ['eaMuPlusLambda']

# Initialise the game environment for the chosen settings
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  speed="fastest",
                  timeexpire=max_budget)

# -------------------------------- Functions --------------------------------- #

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

# write statistics
def write_stats_in_file(stats, file):
    f = open(file, "w+")
    for stat in stats:
        f.write(json.dumps(stat) + "\n")

# Write best individuals in file
def write_best(individuals, folder, exp_name, alg, enemy):
    for count, individual in enumerate(individuals):
        run = count + 1
        with open(f"{folder}exp-{exp_name}_alg-{alg}_enemy-{enemy}_run-{run}.txt", "w+") as f:
            for weight in individual:
                f.write(str(weight) + "\n")

# Test the best individual per run and save resulting statistics 
def eval_best(individuals, folder, exp_name, alg, enemy):
    print("-- Evaluating Best Individuals --")
    # Iterate over the best individuals from each run
    avg_results = []
    for ind in individuals:
        # Test each individual 5 times
        tests = []
        for _ in range(5):
            tests.append(simulation(ind))
        
        # Take the average and append to results list
        avg_results.append(np.mean(np.array(tests), axis=0))
    
    # Convert results to a dataframe
    avg_results_df = pd.DataFrame(avg_results, columns=['Fitness', 'Player Life', 'Enemy Life', 'Time'])
    
    # Save results for future plotting
    avg_results_df.to_csv(f"{folder}exp-{exp_name}_alg-{alg}_enemy-{enemy}.csv")

    # Generate a box plot for each simulation metric
    # NOTE - This plot will not be used directly in the report as it needs to
    # be grouped with the other box plots
    fig, ax = plt.subplots()
    ax.set_title(f"alg-{alg}_enemy-{enemy}")
    ax.set_ylabel('Fitness')
    ax.boxplot(avg_results_df['Fitness'], patch_artist=True, labels=[f"{alg}"])
    ax.yaxis.grid()
    fig.savefig(f"{folder}exp-{exp_name}_alg-{alg}_enemy-{enemy}.pdf")
    plt.close()


def plot_exp_stats(statistics, folder, exp_name, alg, enemy):
    # Combine stats from all runs
    df_stat = pd.concat(statistics)

    # Save results for future plotting
    df_stat.to_csv(f"{folder}exp-{exp_name}_alg-{alg}_enemy-{enemy}.csv")

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

    # Generate line plot
    # NOTE - Generate plots WITHOUT title since we will add a title in the report 
    fig, ax = plt.subplots()
    ax.set_title(f"{alg} Enemy {enemy} - Mean and Maximum Fitness per Generation\n{exp_name}")
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.plot(gen, avg_mean, '-', label='average mean')
    ax.fill_between(gen, avg_mean_minus_std, avg_mean_plus_std, alpha=0.2)
    ax.plot(gen, avg_max, '-', label='average max')
    ax.fill_between(gen, avg_max_minus_std, avg_max_plus_std, alpha=0.2)
    ax.legend(loc='lower right', prop={'size': 15})
    ax.set_ylim(0, 100)
    ax.grid()
    fig.savefig(f"{folder}exp-{exp_name}_alg-{alg}_enemy-{enemy}.png")
    plt.close()
    # plt.show()

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

if crossover == 'cxUniform':
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
elif crossover == 'cxSimulatedBinary':
    toolbox.register("mate", tools.cxSimulatedBinary, eta=2)
else:
    toolbox.register("mate", tools.cxTwoPoint) # crossover operator

if mutation == 'mutGaussian':
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.5)
elif mutation == 'mutFlipBit':
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)
elif mutation == 'mutUniformInt':
    toolbox.register("mutate", tools.mutUniformInt, low=-1, up=1, indpb=0.5)
else:
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

if selection == 'selRandom':
    toolbox.register("select", tools.selRandom)
elif selection == 'selRoulette':
    toolbox.register("select", tools.selRoulette)
else:
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
                pop = toolbox.population(n=MU)
                hof = tools.ParetoFront(eq_)

                print("Start of evolution enemy %i run %i" % (enemy, run))
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

            # Write best individuals fitness values for enemy and experiment
            exp_name = '_'.join([selection, mutation, crossover, str(MU), str(LAMBDA), str(NGEN), str(CXPB), str(MUTPB)])
            plot_exp_stats(statistics, experiment_name + "/plots/", exp_name, alg, enemy)
            write_best(best_individuals, experiment_name + "/best_results/best_weights/", exp_name, alg, enemy)
            eval_best(best_individuals, experiment_name + "/best_results/best_individuals/", exp_name, alg, enemy)

        # Write statistics for experiment
        #write_stats_in_file(log,"log_stats_" + experiment_name + alg + ".txt")

if __name__ == "__main__":
    # Use multiprocessing map implementation for parallelization
    pool = mp.Pool(processes=mp.cpu_count())
    toolbox.register('map', pool.map)

    main()

    pool.close()
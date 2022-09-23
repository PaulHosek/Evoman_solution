# --------------------- Import Frameworks and Libraries --------------------- #
import sys
import os
import json
import time

import yaml
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from deap import tools, creator, base, algorithms

# TODO - What does this option do?
if os.environ.get("EVOMAN_FAST"):
    print("\nUsing evoman_fast!!! ...vrooom\n")
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evoman_fast")
    )
else:
    print("\nUsing standard evoman\n")
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evoman")
    )

from evoman.environment import Environment
from demo_controller import player_controller

# ---------------------------------- Setup ---------------------------------- #

# CONSTANTS
AGENT_LIFE = 100
ENEMY_MODE = "static"
DIFFICULTY = 2

# Load settings
with open("config.yaml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

# Prevent graphics and audio rendering to speed up simulations
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"  # TODO - Does this do anything?
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# Create experiment folder if needed
exp_name = config["exp_name"]
if not os.path.exists(exp_name):
    os.makedirs(exp_name)
    os.makedirs(exp_name + "/best_results")
    os.makedirs(exp_name + "/plots")

# Initialise the game environment for the chosen settings
env = Environment(
    experiment_name=config["exp_name"],
    enemies=config["enemies"],
    player_controller=player_controller(config["n_hidden_neurons"]),
    enemymode=ENEMY_MODE,
    level=DIFFICULTY,
    speed=config["speed"],
    timeexpire=config["timeexpire"],
)

# -------------------------------- Functions -------------------------------- #


# a game simulation for environment env and game x
def simulation(env, x):
    fitness, player_life, enemy_life, game_time = env.play(pcont=x)
    return fitness


# evaluation of game
def cust_evaluate(x):
    sim = simulation(env, x)
    return (sim,)


# Determine individuals that need to be evaluated
def evaluate_pop(env, pop):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [indiv for indiv in pop if not indiv.fitness.valid]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
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


def plot_exp_stats(enemy, statistics, alg):
    # TO-DO:
    # Save statistics before plotting in file so we can play more with the experiments data
    x = range(1, config["n_gens"] + 1)
    means = np.transpose(
        np.mean(statistics, axis=0)
    )  # this migjt be automatically done with seaborn
    # stds = np.transpose(np.std(statistics, axis=0))

    plt.figure(figsize=(10, 8))
    plt.title(
        "%s Enemy %i - Average and Maximum Fitness of each Generation" % (alg, enemy)
    )
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.plot(x, means[0], color="red", label="Mean Fitness")
    plt.plot(x, means[1], color="blue", label="Maximum Fitness")
    plt.legend(loc="lower right")
    plt.savefig(config["exp_name"] + "/plots/" + alg + "_enemy" + str(enemy) + ".png")
    plt.ylim(0, 100)
    plt.show()


# TO-DO: simplify this and call the existing DEAP functions
def create_next_generation(env, pop, alg="eaSimple"):
    # mu and lambda are pop_size
    if alg == "eaSimple":
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Vary the pool of individuals
        algorithms.eaSimple()
        offspring = algorithms.varAnd(
            offspring, toolbox, config["mating_pb"], config["mutation_pb"]
        )
        # Evaluate the individuals with an invalid fitness
        offspring = evaluate_pop(env, offspring)
        # Replace the current population by the offspring
        pop[:] = offspring
    elif alg == "eaMuPlusLambda":
        # Vary the population
        offspring = algorithms.varOr(
            pop,
            toolbox,
            config["pop_size"],
            config["crossover_pb"],
            config["mutation_pb"],
        )
        # Evaluate the individuals with an invalid fitness
        offspring = evaluate_pop(env, offspring)
        # Select the next generation population
        pop[:] = toolbox.select(pop + offspring, config["pop_size"])
    elif alg == "eaMuCommaLambda":
        # Vary the population
        offspring = algorithms.varOr(
            pop,
            toolbox,
            config["pop_size"],
            config["crossover_pb"],
            config["mutation_pb"],
        )
        # Evaluate the individuals with an invalid fitness
        offspring = evaluate_pop(env, offspring)
        # Select the next generation population
        pop[:] = toolbox.select(offspring, config["pop_size"])

    # get best results
    best = tools.HallOfFame(1, similar=np.array_equal)
    return pop, best


# ----------------------------- Initialise DEAP ------------------------------ #

# deap - creating types: Fitness, Individual and Population
# Fitness - tuple, we give one for single objective, 1.0 for maximizing
creator.create("FitnessBest", base.Fitness, weights=(1.0,))
# Individual class inherited from numpy.ndarray
creator.create(
    "Individual",
    np.ndarray,
    fitness=creator.FitnessBest,
    player_life=AGENT_LIFE,
    enemy_life=AGENT_LIFE,
)

# OPERATORS & INITIALIZATIONS OF CLASSES
toolbox = base.Toolbox()
# population drawn from uniform distribution
toolbox.register("indices", np.random.uniform, -1, 1)

# Register functions
toolbox.register("evaluate", cust_evaluate)
toolbox.register("mate", tools.cxTwoPoint)  # crossover operator
toolbox.register(
    "mutate", tools.mutShuffleIndexes, indpb=0.5
)  # TODO: Add to config file
toolbox.register("select", tools.selTournament, tournsize=config["tournsize"])

# Initialize deap logbook
log = tools.Logbook()
log.header = [
    "enemy",
    "run",
    "generation",
    "population size",
    "fitness mean",
    "fitness max",
]

# ---------------------------------- Main ------------------------------------ #


def main():
    for alg in config["algorithms"]:
        print("Algorithm: %s" % alg)
        # For each of the n enemies we want to run the experiment for:
        for enemy in config["enemies"]:
            # number of weights for multilayer network with n_hidden_neurons
            n_weights = (env.get_num_sensors() + 1) * config["n_hidden_neurons"] + (
                config["n_hidden_neurons"] + 1
            ) * 5
            toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                toolbox.indices,
                n=n_weights,
            )
            toolbox.register(
                "population",
                tools.initRepeat,
                list,
                toolbox.individual,
                n=config["pop_size"],
            )

            best_individuals = []
            statistics = []

            # We run the experiment a few times - n_runs
            for run in range(1, config["n_runs"] + 1):
                start_time = time.time()
                gen_stat = []

                # ---------------Initialize population ---------------
                # pop = np.random.uniform(low=-1, high=1, size=(n_population, n_weights))
                pop = toolbox.population(n=config["pop_size"])
                # evaluate first generation
                pop = evaluate_pop(env, pop)
                best = tools.HallOfFame(1, similar=np.array_equal)
                # record these best results in file
                stat = record_stat(pop, generation=0, run=run, enemy=enemy, best=best)
                gen_stat.append(stat)

                print("Start of evolution player %i run %i" % (enemy, run))
                for generation in range(1, config["n_gens"]):
                    pop, best = create_next_generation(env, pop, alg)
                    # record these best results in file
                    stat = record_stat(
                        pop, generation=generation, run=run, enemy=enemy, best=best
                    )
                    gen_stat.append(stat)

                print("-- End of (successful) evolution --")
                statistics.append(gen_stat)
                best_individuals.append(best[0])
                print("---- %s seconds elapsed ----" % (time.time() - start_time))

            # Write best individuals fitness values for enemy and experiment
            write_best(
                best_individuals,
                config["exp_name"]
                + "/best_results/Best_individuals_"
                + config["exp_name"]
                + alg,
                enemy,
            )
            plot_exp_stats(enemy, statistics, alg)

        # Write statistics for experiment
        write_stats_in_file(log, "log_stats_" + config["exp_name"] + alg + ".txt")


if __name__ == "__main__":
    # Replace DEAP map with multiprocessing for parallelization
    pool = mp.Pool(processes=mp.cpu_count())
    toolbox.register("map", pool.map)

    main()

    pool.close()

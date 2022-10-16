import sys, os, glob

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
# imports other libs
import numpy as np
import matplotlib.pyplot as plt
import csv

experiment_name = 'controller_generalist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# Prevent graphics and audio rendering to speed up simulations
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")


def find_best_weights(f_prefix_path, n_solutions, n_runs=1, txt_csv=".txt"):
    n_solutions = len(glob.glob1(f_prefix_path,"*.txt"))
    all_ea = np.empty((n_solutions, 265))
    idx = 0
    algs = []
    for file in os.listdir(f_prefix_path):
        if file.endswith(".txt"):
            all_ea[idx, :] = np.loadtxt(f_prefix_path + file)
            print(f"{idx} - {file}")
            idx= idx+1
            algs.append(file)

    best_in_fit = {"Index Weight": 0, "Alg Name": "", "Value": 0, "Weights": []}
    best_in_gain = {"Index Weight": 0, "Alg Name": "", "Value": 0, "Weights": []}
    best_in_defeated = {"Index Weight": 0, "Alg Name": "", "Value": 0, "Weights": []}
    for idx, cur_weights in enumerate(all_ea):
        print(f"Evaluating {idx}")
        # (re-)set performance measures
        fitnesses = np.empty((n_runs, 8))
        gains = np.empty((n_runs, 8))
        n_defeated = 0
        for run in range(n_runs):
            print(f"--- run {run}")
            cur_result = np.empty((8, 4))
            for en in range(1, 9):
                # Update the enemy
                env.update_parameter('enemies', [en])

                cur_result[en - 1] = env.play(cur_weights)
                # if player defeats enemy
                if cur_result[en - 1, 1] - cur_result[en - 1, 2] > 0: n_defeated += 1
            # results of all enemies for that run
            fitnesses[run, :] = cur_result[:, 0]
            gains[run, :] = cur_result[:, 1] - cur_result[:, 2]

        # flatten over runs
        fitnesses = np.mean(fitnesses, axis=0)
        gains = np.mean(fitnesses, axis=0)

        mean_fitness = np.mean(fitnesses) - np.std(fitnesses)
        gain = np.mean(gains)

        if best_in_fit["Value"] < mean_fitness:
            best_in_fit.update({'Index Weight': idx, 'Alg Name': algs[idx], 'Value': mean_fitness, 'Weights': cur_weights})

        if best_in_gain["Value"] < gain:
            best_in_gain.update({'Index Weight': idx, 'Alg Name': algs[idx], 'Value': gain, 'Weights': cur_weights})

        if best_in_defeated["Value"] < n_defeated:
            best_in_defeated.update({'Index Weight': idx, 'Alg Name': algs[idx], 'Value': n_defeated, 'Weights': cur_weights})

    return {"Fitness": best_in_fit, "Gain": best_in_gain, "Defeated": best_in_defeated}


# tests saved demo solutions for each enemy
def assess_per_enemy(weights, n_runs=10, full_out = False):
    if full_out:
        mean_values = np.empty((n_runs, 8, 4))
    fitnesses = np.empty((n_runs ,8))
    gains = np.empty((n_runs ,8))
    for run in range(n_runs):
        cur_result = np.empty((8, 4))
        for en in range(1, 9):
            # Update the enemy
            env.update_parameter('enemies', [en])

            cur_result[en-1] = env.play(weights)
        if full_out: mean_values[run,:,:] = cur_result

        fitnesses[run, :] = cur_result[:,0]
        gains[run, :] = cur_result[:,1] - cur_result[:,2]
    if full_out: return fitnesses,gains,np.mean(mean_values, axis=0)
    return fitnesses, gains


# path is universal prefix of all algorithms without the number at the end e.g., mo-cma-1.txt -> mo-cma-
#main_folder = "task2/exp_results_all/"
main_folder = "task2/exp_results/"
path = main_folder + "best_results/best_weights/"

sol = find_best_weights(f_prefix_path=path, n_solutions=8, txt_csv=".txt", n_runs=1)
print(sol['Fitness']['Alg Name'])
print(sol['Gain']['Alg Name'])
print(sol['Defeated']['Alg Name'])
_, _, full_out = assess_per_enemy(weights=sol['Defeated']['Weights'], n_runs=1, full_out=True)
print(full_out)

with open(main_folder + 'best_fitness.csv', 'w+') as csvfile:
    for key in sol['Fitness'].keys():
        csvfile.write("%s, %s\n" % (key, sol['Fitness'][key]))

with open(main_folder + 'best_gain.csv', 'w+') as csvfile:
    for key in sol['Gain'].keys():
        csvfile.write("%s, %s\n" % (key, sol['Gain'][key]))

with open(main_folder + 'best__defeated_enemies.csv', 'w+') as csvfile:
    for key in sol['Defeated'].keys():
        csvfile.write("%s, %s\n" % (key, sol['Defeated'][key]))

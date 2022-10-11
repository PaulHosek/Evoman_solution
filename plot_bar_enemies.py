import sys, os

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
# imports other libs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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




# example weights
# change this for testing
weights = np.loadtxt('ea_exp/best_results/Best_individuals_ea_expeaMuPlusLambda_e1_run2.txt')
weights_2 = np.loadtxt('ea_exp/best_results/Best_individuals_ea_expeaMuPlusLambda_e6_run4.txt')


# tests saved demo solutions for each enemy
def assess_per_enemy(weights, n_runs=10):
    fitnesses = np.empty((n_runs ,8))
    gains = np.empty((n_runs ,8))
    for run in range(n_runs):
        cur_result = np.empty((8, 4))
        for en in range(1, 9):
            # Update the enemy
            env.update_parameter('enemies', [en])

            cur_result[en-1] = env.play(weights)
        fitnesses[run, :] = cur_result[:,0]
        gains[run, :] = cur_result[:,1] - cur_result[:,2]
    return fitnesses, gains


# Plotting
def plot_measure(data:list,measure_name,file_name, labels=["EA 1", "EA 2"], show_save="show"):
    """
    Plot fitness or gain from function "assess per enemy"
    @param data: list of 2d gains/fitnesses for different algorithms
    @param measure_name:
    @return:
    """

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    colors = ["blue", "green","red"]
    for idx, cur_data in enumerate(data):
        bp = ax.boxplot(cur_data, patch_artist=True)

        # visual adjustments to box plots
        for box in bp["boxes"]:
            box.set( color=colors[idx], linewidth=2, alpha =0.6)
            box.set( facecolor = colors[idx])
        for i in bp['whiskers']:
            i.set(color=colors[idx], linewidth=2, alpha =0.6)
        for median in bp['medians']:
            median.set(color=colors[idx], linewidth=2, alpha =0.6)
        # plot median
        ax.plot(np.arange(1,9,1),np.median(cur_data, axis=0), color=colors[idx], alpha = 0.5,label =labels[idx])


    ax.set_ylabel(measure_name)
    ax.set_xlabel("Enemy")

    if show_save == "save":
        plt.legend(loc="best")
        plt.savefig(file_name, dpi=300)

    elif show_save == "show":
        plt.legend(loc="best")
        plt.show()
    return


fitnesses, gains = assess_per_enemy(weights=weights,n_runs=10)
fitnesses_2, gains_2 = assess_per_enemy(weights=weights_2,n_runs=10)

plot_measure([gains,gains_2], "Gain", 'some_figure_name',show_save="show",labels=["MO example","CMA example"])


import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
import json
import optuna
from optuna.trial import TrialState

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def simulation(env, x):
    results = env.play(pcont=x)
    return results

def plot_exp_stats(statistics, folder, exp_name, alg, enemy, ngen):
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

    gen = range(0, ngen)

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

def save_exp(population, folder, exp_name, alg, enemy, run):
    with open(f"{folder}exp-{exp_name}_alg-{alg}_enemy-{enemy}_run-{run}.pickle", "wb") as f:
        pickle.dump(population, f)



# Test the best individual per run and save resulting statistics
def eval_best(env, individuals, folder, exp_name, alg, enemy):
    print("-- Evaluating Best Individuals --")
    # Iterate over the best individuals from each run
    avg_results = []
    for ind in individuals:
        # Test each individual 5 times
        tests = []
        for _ in range(5):
            tests.append(simulation(env, ind))

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

# --------------------------------- Tuning ---------------------------------- #

# Save study as a backup
def save_study(study, folder):
    with open(f"{folder}/study.pkl", "wb") as f:
        pickle.dump(study, f)

# Load saved study
def load_study(folder):
    with open(f"{folder}/study.pkl", "rb") as f:
        study = pickle.load(f)
        return study

# Print tuning results to the console
def print_tuning_summary(study, settings):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    best_trial = study.best_trial

    print()
    print("------- EXP SETTINGS -------")
    print()
    for key, value in settings.items():
        print(f"{key}: {value}")
    print()
    print('------- TUNING RESULTS -------')
    print()
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    print()
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print()
    print('------------------------------')
    print()


# Save tuning results to a text file
def save_tuning_summary(study, folder, settings):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    best_trial = study.best_trial

    settings_str = ["------- EXP SETTINGS -------\n"] + [f"{key}: {value}" for key, value in settings.items()] + ["\n"]

    results_str = [
        "------- TUNING RESULTS -------\n",
        "Study statistics: ",
        f"  Number of finished trials: {len(study.trials)}",
        f"  Number of pruned trials: {len(pruned_trials)}",
        f"  Number of complete trials: {len(complete_trials)}\n",
        "Best trial: ",
        f"  Value: {best_trial.value}",
        "  Params: "
    ]

    best_params_str = [f"    {key}: {value}" for key, value in best_trial.params.items()]
    results_str = results_str + best_params_str

    lines = settings_str + results_str

    with open(f"{folder}/summary.txt", "w") as f:
        f.write("\n".join(lines))

# Plot and save tuning visualization plots
def save_tuning_plots(study, folder):
    if not optuna.visualization.is_available():
        print("Plotly visualization is unavailable. Plotly version >4.0.0 required.")
        return

    print("Saving plots...")

    fig = optuna.visualization.plot_contour(study)
    fig.write_image(f"{folder}/contour.pdf")

    fig = optuna.visualization.plot_edf(study)
    fig.write_image(f"{folder}/edf.pdf")

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.write_image(f"{folder}/intermediate_values.pdf")

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"{folder}/optimization_history.pdf")

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(f"{folder}/parallel_coordinate.pdf")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f"{folder}/param_importances.pdf")

    fig = optuna.visualization.plot_slice(study)
    fig.write_image(f"{folder}/slice.pdf")

    print("Done!")
    
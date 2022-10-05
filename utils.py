import matplotlib.pyplot as plt
import pandas as pd


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

    gen = range(0, ngen + 1)

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
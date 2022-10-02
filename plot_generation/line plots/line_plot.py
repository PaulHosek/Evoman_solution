import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------- Settings --------------------------------- #

alpha = 0.15
linewidth = 2
legend_size = 18
axis_font_size = 14
tick_label_size = 13
axes_aspect_ratio = 1.0

# ------- EA 1 ------- #
filename1 = "exp-selTournament_mutUniformInt_cxBlend_100_100_100_0.8_0.2_alg-eaMuPlusLambda_enemy-7.csv"
mean_colour1 = '#d62728'
max_colour1 = '#ff7f0e'

# ------- EA 2 ------- #
filename2 = "exp-selTournament_mutUniformInt_cxSimulatedBinary_100_100_100_0.8_0.2_alg-eaMuPlusLambda_enemy-7.csv"
mean_colour2 = '#1f77b4'
max_colour2 = '#17becf'

# -------------------------- Read and Prepare Data -------------------------- #

# ------- EA 1 ------- #
df_stat1 = pd.read_csv("line_plot_data/" + filename1)

# Avg and std of means
avg_mean1 = df_stat1.groupby(['gen'])['mean'].mean().to_numpy()
std_mean1 = df_stat1.groupby(['gen'])['mean'].std().to_numpy()
avg_mean_plus_std1 = [a + b for a, b in zip(avg_mean1, std_mean1)]
avg_mean_minus_std1 = [a - b for a, b in zip(avg_mean1, std_mean1)]

# Avg and std of maxes
avg_max1 = df_stat1.groupby(['gen'])['max'].mean().to_numpy()
std_max1 = df_stat1.groupby(['gen'])['max'].std().to_numpy()
avg_max_plus_std1 = [a + b for a, b in zip(avg_max1, std_max1)]
avg_max_minus_std1 = [a - b for a, b in zip(avg_max1, std_max1)]

# ------- EA 2 ------- #
df_stat2 = pd.read_csv("line_plot_data/" + filename2)

# Avg and std of means
avg_mean2 = df_stat2.groupby(['gen'])['mean'].mean().to_numpy()
std_mean2 = df_stat2.groupby(['gen'])['mean'].std().to_numpy()
avg_mean_plus_std2 = [a + b for a, b in zip(avg_mean2, std_mean2)]
avg_mean_minus_std2 = [a - b for a, b in zip(avg_mean2, std_mean2)]

# Avg and std of maxes
avg_max2 = df_stat2.groupby(['gen'])['max'].mean().to_numpy()
std_max2 = df_stat2.groupby(['gen'])['max'].std().to_numpy()
avg_max_plus_std2 = [a + b for a, b in zip(avg_max2, std_max2)]
avg_max_minus_std2 = [a - b for a, b in zip(avg_max2, std_max2)]

gen = range(0, len(avg_mean1))

# ------------------------------ Plot the Data ------------------------------ #

fig, (ax1, ax2) = plt.subplots(1, 2)

# ------- Means ------- #
# EA 1
ax1.plot(gen, avg_mean1, '-', linewidth=linewidth, label=r'BLX-$\alpha$', color=mean_colour1)
ax1.fill_between(gen, avg_mean_minus_std1, avg_mean_plus_std1, alpha=alpha, color=mean_colour1)

# EA 2
ax1.plot(gen, avg_mean2, '-', linewidth=linewidth, label='SBX', color=mean_colour2)
ax1.fill_between(gen, avg_mean_minus_std2, avg_mean_plus_std2, alpha=alpha, color=mean_colour2)

# Labels
ax1.set_title('Means', fontsize=axis_font_size)
ax1.set_xlabel('Generation', fontsize=axis_font_size)
ax1.set_ylabel('Fitness', fontsize=axis_font_size)
ax1.legend(loc='lower right', prop={'size': legend_size})

# Formatting
ax1.margins(x=0)
ax1.set_ylim(0, 100)
ax1.tick_params(axis='x', labelsize=tick_label_size)
ax1.tick_params(axis='y', labelsize=tick_label_size)
ax1.grid()
# x_left, x_right = ax1.get_xlim()
# y_low, y_high = ax1.get_ylim()
# ax1.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * axes_aspect_ratio)

# ------- Maxes ------- #

# EA 1
ax2.plot(gen, avg_max1, '-', linewidth=linewidth, label=r'BLX-$\alpha$', color=max_colour1)
ax2.fill_between(gen, avg_max_minus_std1, avg_max_plus_std1, alpha=alpha, color=max_colour1)

# EA 2
ax2.plot(gen, avg_max2, '-', linewidth=linewidth, label='SBX', color=max_colour2)
ax2.fill_between(gen, avg_max_minus_std2, avg_max_plus_std2, alpha=alpha, color=max_colour2)

# Labels
ax2.set_title('Maximums', fontsize=axis_font_size)
ax2.set_xlabel('Generation', fontsize=axis_font_size)
ax2.legend(loc='lower right', prop={'size': legend_size})

# Formatting
ax2.margins(x=0)
ax2.set_ylim(0, 100)
ax2.tick_params(axis='x', labelsize=tick_label_size)
ax2.tick_params('y', labelleft=False)
ax2.grid()
# x_left, x_right = ax2.get_xlim()
# y_low, y_high = ax2.get_ylim()
# ax2.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * axes_aspect_ratio)

# Figure adjustments
fig.set_size_inches(10, 5)
fig.subplots_adjust(left=0.075, bottom=0.11, right=0.98, top=0.937, wspace=0.095, hspace=0.2)
fig.tight_layout()

# plt.show()

# Save figures
enemy_num = filename1[-5]
if enemy_num != filename2[-5]:
    print("ERROR: Enemy Mismatch!")
    exit()

fig.savefig(f"line_plots/enemy {enemy_num}.pdf")
plt.close()
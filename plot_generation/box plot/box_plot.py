import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------- Settings --------------------------------- #

# ------- EA 1 ------- #
filename1A = "exp-selTournament_mutUniformInt_cxBlend_150_150_100_0.8_0.2_alg-eaMuPlusLambda_enemy-2.csv"
filename1B = "exp-selTournament_mutUniformInt_cxBlend_150_150_100_0.8_0.2_alg-eaMuPlusLambda_enemy-3.csv"
filename1C = "exp-selTournament_mutUniformInt_cxBlend_150_150_100_0.8_0.2_alg-eaMuPlusLambda_enemy-6.csv"

# ------- EA 2 ------- #
filename2A = "exp-selTournament_mutUniformInt_cxSimulatedBinary_150_150_100_0.8_0.2_alg-eaMuPlusLambda_enemy-2.csv"
filename2B = "exp-selTournament_mutUniformInt_cxSimulatedBinary_150_150_100_0.8_0.2_alg-eaMuPlusLambda_enemy-3.csv"
filename2C = "exp-selTournament_mutUniformInt_cxSimulatedBinary_150_150_100_0.8_0.2_alg-eaMuPlusLambda_enemy-6.csv"

color1 = '#F9ACB1'
color2 = '#96C4DB'

box_width = 0.6
axis_font_size = 18
legend_size = 18

# -------------------------- Read and Prepare Data -------------------------- #

p_values = ['0.1', '0.2', '0.3']

# ------- EA 1 ------- #
ea1_enemyA = pd.read_csv("box_plot_data/" + filename1A)
ea1_enemyB = pd.read_csv("box_plot_data/" + filename1B)
ea1_enemyC = pd.read_csv("box_plot_data/" + filename1C)

# ------- EA 2 ------- #
ea2_enemyA = pd.read_csv("box_plot_data/" + filename2A)
ea2_enemyB = pd.read_csv("box_plot_data/" + filename2B)
ea2_enemyC = pd.read_csv("box_plot_data/" + filename2C)

# Calculate the Individual Gains
ea1_enemyA = ea1_enemyA["Player Life"] - ea1_enemyA["Enemy Life"]
ea1_enemyB = ea1_enemyB["Player Life"] - ea1_enemyB["Enemy Life"]
ea1_enemyC = ea1_enemyC["Player Life"] - ea1_enemyC["Enemy Life"]

ea2_enemyA = ea2_enemyA["Player Life"] - ea2_enemyA["Enemy Life"]
ea2_enemyB = ea2_enemyB["Player Life"] - ea2_enemyB["Enemy Life"]
ea2_enemyC = ea2_enemyC["Player Life"] - ea2_enemyC["Enemy Life"]

ea1 = [ea1_enemyA.tolist(), ea1_enemyB.tolist(), ea1_enemyC.tolist()]
ea2 = [ea2_enemyA.tolist(), ea2_enemyB.tolist(), ea2_enemyC.tolist()]

# # Uncomment when not comparing individual gain
# ea1 = [ea1_enemyA[key].tolist(), ea1_enemyB[key].tolist(), ea1_enemyC[key].tolist()]
# ea2 = [ea2_enemyA[key].tolist(), ea2_enemyB[key].tolist(), ea2_enemyC[key].tolist()]

# ------------------------------ Plot the Data ------------------------------ #

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def set_face_color(bp, color):
    for box in bp['boxes']:
        box.set_facecolor = color

def set_median_color(bp, color):
    for median in bp['medians']:
        median.set_color(color)

plt.figure()

bplot1 = plt.boxplot(ea1, positions=np.array(range(len(ea1)))*2.0-0.4, widths=box_width, patch_artist=True)
bplot2 = plt.boxplot(ea2, positions=np.array(range(len(ea2)))*2.0+0.4, widths=box_width, patch_artist=True)

for box in bplot1['boxes']:
    box.set_facecolor(color1)

for box in bplot2['boxes']:
    box.set_facecolor(color2)

for median in bplot1['medians']:
    median.set_color('black')

for median in bplot2['medians']:
    median.set_color('black')

# Use dummy lines to create a legend
plt.plot([], c=color1, linewidth=6, label=r'BLX-$\alpha$')
plt.plot([], c=color2, linewidth=6, label='SBX')
plt.legend(loc='lower left', prop={'size': legend_size}, borderpad=0.25)

# Formatting
enemy_nums = [filename1A[-5], filename1B[-5], filename1C[-5],]
ticks = [f"Enemy {enemy_nums[0]}", f"Enemy {enemy_nums[1]}", f"Enemy {enemy_nums[2]}"]
plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=axis_font_size)

plt.xlim(-1, len(ticks)*2 - 1)
# plt.ylim(83, 94)
plt.ylabel("Individual Gain", fontsize = axis_font_size)
plt.grid(axis = 'y')
plt.tight_layout()

# plt.show()

# Ensure correct file combination is being used
if filename1A[-5] != filename2A[-5] or filename1B[-5] != filename2B[-5] or filename1C[-5] != filename2C[-5]:
    print("ERROR: Enemy Mismatch!")
    exit()

plt.savefig(f"box_plots/boxplot_enemies_{enemy_nums[0]}-{enemy_nums[1]}-{enemy_nums[2]}.pdf")
plt.close()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def smooth(x, timestamps=9):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
color_cycle = sns.color_palette()
sns.set_color_codes()

colors = [0, 5, 2, 6, 1, 3]

alpha = 0.3
legend_size = 15
line_size_others = 2
line_size_ours = 2
tick_size = 18
label_size = 18

episode_rewards = '/eval_rewards.npy'
obs_window = 100

X = np.arange(obs_window)
"""easy mode"""
easy_maa2c_0 = np.load('../results/Jan-02_09_02_38' + episode_rewards)
easy_maa2c_2000 = np.load('../results/Jan-02_09_02_53' + episode_rewards)
easy_maa2c_2021 = np.load('../results/Jan-02_09_03_05' + episode_rewards)
easy_maa2c = np.vstack((smooth(easy_maa2c_0), smooth(easy_maa2c_2000), smooth(easy_maa2c_2021)))

easy_maa2c_mean = np.mean(easy_maa2c, axis=0)
easy_maa2c_std = np.std(easy_maa2c, axis=0)
easy_maa2c_lower_bound = easy_maa2c_mean - easy_maa2c_std
easy_maa2c_upper_bound = easy_maa2c_mean + easy_maa2c_std

##############################
easy_maacktr_0 = np.load('../results/Dec-31_22_38_28' + episode_rewards)
easy_maacktr_2000 = np.load('../results/Dec-31_22_38_56' + episode_rewards)
easy_maacktr_2021 = np.load('../results/Dec-31_22_39_14' + episode_rewards)
easy_maacktr = np.vstack((smooth(easy_maacktr_0), smooth(easy_maacktr_2000), smooth(easy_maacktr_2021)))

easy_maacktr_mean = np.mean(easy_maacktr, axis=0)
easy_maacktr_std = np.std(easy_maacktr, axis=0)
easy_maacktr_lower_bound = easy_maacktr_mean - easy_maacktr_std
easy_maacktr_upper_bound = easy_maacktr_mean + easy_maacktr_std

##############################
easy_mappo_0 = np.load('../results/Jan-01_11_21_56' + episode_rewards)
easy_mappo_2000 = np.load('../results/Jan-01_11_22_31' + episode_rewards)
easy_mappo_2021 = np.load('../results/Jan-01_11_22_43' + episode_rewards)
easy_mappo = np.vstack((smooth(easy_mappo_0), smooth(easy_mappo_2000), smooth(easy_mappo_2021)))

easy_mappo_mean = np.mean(easy_mappo, axis=0)
easy_mappo_std = np.std(easy_mappo, axis=0)
easy_mappo_lower_bound = easy_mappo_mean - easy_mappo_std
easy_mappo_upper_bound = easy_mappo_mean + easy_mappo_std

##############################
easy_ours_0 = np.load('../results/Dec-30_14_34_59' + episode_rewards)
easy_ours_2000 = np.load('../results//Dec-30_14_35_21' + episode_rewards)
easy_ours_2021 = np.load('../results/Dec-30_14_35_41' + episode_rewards)
easy_ours = np.vstack((smooth(easy_ours_0), smooth(easy_ours_2000), smooth(easy_ours_2021)))

easy_ours_mean = np.mean(easy_ours, axis=0)
easy_ours_std = np.std(easy_ours, axis=0)
easy_ours_lower_bound = easy_ours_mean - easy_ours_std
easy_ours_upper_bound = easy_ours_mean + easy_ours_std

"""medium mode"""
medium_maa2c_0 = np.load('../results/Jan-02_18_58_17' + episode_rewards)
medium_maa2c_2000 = np.load('../results/Jan-02_18_58_29' + episode_rewards)
medium_maa2c_2021 = np.load('../results/Jan-02_18_58_41' + episode_rewards)
medium_maa2c = np.vstack((smooth(medium_maa2c_0), smooth(medium_maa2c_2000), smooth(medium_maa2c_2021)))

medium_maa2c_mean = np.mean(medium_maa2c, axis=0)
medium_maa2c_std = np.std(medium_maa2c, axis=0)
medium_maa2c_lower_bound = medium_maa2c_mean - medium_maa2c_std
medium_maa2c_upper_bound = medium_maa2c_mean + medium_maa2c_std

medium_maacktr_0 = np.load('../results/Dec-31_23_07_18' + episode_rewards)
medium_maacktr_2000 = np.load('../results/Dec-31_23_19_01' + episode_rewards)
medium_maacktr_2021 = np.load('../results/Dec-31_23_19_22' + episode_rewards)
medium_maacktr = np.vstack((smooth(medium_maacktr_0), smooth(medium_maacktr_2000), smooth(medium_maacktr_2021)))

medium_maacktr_mean = np.mean(medium_maacktr, axis=0)
medium_maacktr_std = np.std(medium_maacktr, axis=0)
medium_maacktr_lower_bound = medium_maacktr_mean - medium_maacktr_std
medium_maacktr_upper_bound = medium_maacktr_mean + medium_maacktr_std

##############################
medium_mappo_0 = np.load('../results/Jan-02_09_07_15' + episode_rewards)
medium_mappo_2000 = np.load('../results/Jan-02_09_07_40' + episode_rewards)
medium_mappo_2021 = np.load('../results/Jan-02_09_08_22' + episode_rewards)
medium_mappo = np.vstack((smooth(medium_mappo_0), smooth(medium_mappo_2000), smooth(medium_mappo_2021)))

medium_mappo_mean = np.mean(medium_mappo, axis=0)
medium_mappo_std = np.std(medium_mappo, axis=0)
medium_mappo_lower_bound = medium_mappo_mean - medium_mappo_std
medium_mappo_upper_bound = medium_mappo_mean + medium_mappo_std

##############################
medium_ours_0 = np.load('../results/Dec-30_14_41_13' + episode_rewards)
medium_ours_2000 = np.load('../results/Dec-30_14_41_25' + episode_rewards)
medium_ours_2021 = np.load('../results/Dec-30_14_41_36' + episode_rewards)
medium_ours = np.vstack((smooth(medium_ours_0), smooth(medium_ours_2000), smooth(medium_ours_2021)))

medium_ours_mean = np.mean(medium_ours, axis=0)
medium_ours_std = np.std(medium_ours, axis=0)
medium_ours_lower_bound = medium_ours_mean - medium_ours_std
medium_ours_upper_bound = medium_ours_mean + medium_ours_std


"""hard mode"""
hard_maa2c_0 = np.load('../results/Jan-02_09_03_22' + episode_rewards)
hard_maa2c_2000 = np.load('../results/Jan-02_10_08_16' + episode_rewards)
hard_maa2c_2021 = np.load('../results/Jan-02_10_08_35' + episode_rewards)
hard_maa2c = np.vstack((smooth(hard_maa2c_0), smooth(hard_maa2c_2000), smooth(hard_maa2c_2021)))

hard_maa2c_mean = np.mean(hard_maa2c, axis=0)
hard_maa2c_std = np.std(hard_maa2c, axis=0)
hard_maa2c_lower_bound = hard_maa2c_mean - hard_maa2c_std
hard_maa2c_upper_bound = hard_maa2c_mean + hard_maa2c_std

hard_maacktr_0 = np.load('../results/Jan-01_05_56_04' + episode_rewards)
hard_maacktr_2000 = np.load('../results/Jan-01_05_56_21' + episode_rewards)
hard_maacktr_2021 = np.load('../results/Jan-01_05_56_36' + episode_rewards)
hard_maacktr = np.vstack((smooth(hard_maacktr_0), smooth(hard_maacktr_2000), smooth(hard_maacktr_2021)))

hard_maacktr_mean = np.mean(hard_maacktr, axis=0)
hard_maacktr_std = np.std(hard_maacktr, axis=0)
hard_maacktr_lower_bound = hard_maacktr_mean - hard_maacktr_std
hard_maacktr_upper_bound = hard_maacktr_mean + hard_maacktr_std

##############################
hard_mappo_0 = np.load('../results/Jan-02_00_15_56' + episode_rewards)
hard_mappo_2000 = np.load('../results/Jan-02_00_16_09' + episode_rewards)
hard_mappo_2021 = np.load('../results/Jan-02_00_16_24' + episode_rewards)
hard_mappo = np.vstack((smooth(hard_mappo_0), smooth(hard_mappo_2000), smooth(hard_mappo_2021)))

hard_mappo_mean = np.mean(hard_mappo, axis=0)
hard_mappo_std = np.std(hard_mappo, axis=0)
hard_mappo_lower_bound = hard_mappo_mean - hard_mappo_std
hard_mappo_upper_bound = hard_mappo_mean + hard_mappo_std

##############################
hard_ours_0 = np.load('../results/Dec-31_14_24_18' + episode_rewards)
hard_ours_2000 = np.load('../results/Dec-31_14_24_34' + episode_rewards)
hard_ours_2021 = np.load('../results/Dec-31_14_24_48' + episode_rewards)
hard_ours = np.vstack((smooth(hard_ours_0), smooth(hard_ours_2000), smooth(hard_ours_2021)))

hard_ours_mean = np.mean(hard_ours, axis=0)
hard_ours_std = np.std(hard_ours, axis=0)
hard_ours_lower_bound = hard_ours_mean - hard_ours_std
hard_ours_upper_bound = hard_ours_mean + hard_ours_std

######################################
"""easy mode"""
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(20, 4))
ax0.set_title('Easy Mode', size=label_size)
ax0.plot(X, easy_maa2c_mean[:obs_window], lw=line_size_others, label='MAA2C', linestyle=':',
         color=color_cycle[colors[0]])
ax0.fill_between(X, easy_maa2c_lower_bound[:obs_window], easy_maa2c_upper_bound[:obs_window],
                 facecolor=color_cycle[colors[0]], edgecolor='none', alpha=alpha)

ax0.plot(X, easy_maacktr_mean[:obs_window], lw=line_size_others, label='MAACKTR', linestyle=':', color=color_cycle[colors[1]])
ax0.fill_between(X, easy_maacktr_lower_bound[:obs_window], easy_maacktr_upper_bound[:obs_window], facecolor=color_cycle[colors[1]],
                 edgecolor='none', alpha=alpha)

ax0.plot(X, easy_mappo_mean[:obs_window], lw=line_size_others, label='MAPPO', linestyle=':', color=color_cycle[colors[2]])
ax0.fill_between(X, easy_mappo_lower_bound[:obs_window], easy_mappo_upper_bound[:obs_window], facecolor=color_cycle[colors[2]],
                 edgecolor='none', alpha=alpha)

ax0.plot(X, easy_ours_mean[:obs_window], lw=line_size_ours, label='Ours', color=color_cycle[colors[-1]])
ax0.fill_between(X, easy_ours_lower_bound[:obs_window], easy_ours_upper_bound[:obs_window],
                 facecolor=color_cycle[colors[-1]], edgecolor='none', alpha=alpha)
leg0 = ax0.legend(fontsize=legend_size, loc='lower right', ncol=2)

ax0.set_xlim(0, obs_window)
ax0.set_ylim(10, 80)
ax0.tick_params(axis='x', labelsize=tick_size)
ax0.tick_params(axis='y', labelsize=tick_size)
ax0.set_xlabel('Evaluation epochs', fontsize=label_size)
ax0.set_ylabel('Evaluation reward', fontsize=label_size)
ax0.ticklabel_format(axis="x")

ax0.grid()
# set the linewidth of each legend object
for legobj in leg0.legendHandles:
    legobj.set_linewidth(2.0)

"""medium mode"""
ax1.set_title('Medium Mode', size=label_size)
ax1.plot(X, medium_maa2c_mean[:obs_window], lw=line_size_others, label='MAA2C', linestyle=':',
         color=color_cycle[colors[0]])
ax1.fill_between(X, medium_maa2c_lower_bound[:obs_window], medium_maa2c_upper_bound[:obs_window],
                 facecolor=color_cycle[colors[0]], edgecolor='none', alpha=alpha)

ax1.plot(X, medium_maacktr_mean[:obs_window], lw=line_size_others, label='MAACKTR', linestyle=':',
         color=color_cycle[colors[1]])
ax1.fill_between(X, medium_maacktr_lower_bound[:obs_window], medium_maacktr_upper_bound[:obs_window],
                 facecolor=color_cycle[colors[1]], edgecolor='none', alpha=alpha)

ax1.plot(X, medium_mappo_mean[:obs_window], lw=line_size_others, label='MAPPO', linestyle=':',
         color=color_cycle[colors[2]])
ax1.fill_between(X, medium_mappo_lower_bound[:obs_window], medium_mappo_upper_bound[:obs_window],
                 facecolor=color_cycle[colors[2]], edgecolor='none', alpha=alpha)

ax1.plot(X, medium_ours_mean[:obs_window], lw=line_size_ours, label='Ours', color=color_cycle[colors[-1]])
ax1.fill_between(X, medium_ours_lower_bound[:obs_window], medium_ours_upper_bound[:obs_window],
                 facecolor=color_cycle[colors[-1]], edgecolor='none', alpha=alpha)
leg1 = ax1.legend(fontsize=legend_size, loc='lower right', ncol=2)

ax1.set_xlim(0, obs_window)
ax1.set_ylim(-50, 75)
ax1.tick_params(axis='x', labelsize=tick_size)
ax1.tick_params(axis='y', labelsize=tick_size)
ax1.set_xlabel('Evaluation epochs', fontsize=label_size)
ax1.set_ylabel('Evaluation reward', fontsize=label_size)
ax1.ticklabel_format(axis="x")

ax1.grid()
# set the linewidth of each legend object
for legobj in leg1.legendHandles:
    legobj.set_linewidth(2.0)

"""hard mode"""
ax2.set_title('Hard Mode', size=label_size)
ax2.plot(X, hard_maa2c_mean[:obs_window], lw=line_size_others, label='MAA2C', linestyle=':',
         color=color_cycle[colors[0]])
ax2.fill_between(X, hard_maa2c_lower_bound[:obs_window], hard_maa2c_upper_bound[:obs_window],
                 facecolor=color_cycle[colors[0]], edgecolor='none', alpha=alpha)

ax2.plot(X, hard_maacktr_mean[:obs_window], lw=line_size_others, label='MAACKTR', linestyle=':', color=color_cycle[colors[1]])
ax2.fill_between(X, hard_maacktr_lower_bound[:obs_window], hard_maacktr_upper_bound[:obs_window], facecolor=color_cycle[colors[1]],
                 edgecolor='none', alpha=alpha)

ax2.plot(X, hard_mappo_mean[:obs_window], lw=line_size_others, label='MAPPO', linestyle=':', color=color_cycle[colors[2]])
ax2.fill_between(X, hard_mappo_lower_bound[:obs_window], hard_mappo_upper_bound[:obs_window], facecolor=color_cycle[colors[2]],
                 edgecolor='none', alpha=alpha)

ax2.plot(X, hard_ours_mean[:obs_window], lw=line_size_ours, label='Ours', color=color_cycle[colors[-1]])
ax2.fill_between(X, hard_ours_lower_bound[:obs_window], hard_ours_upper_bound[:obs_window],
                 facecolor=color_cycle[colors[-1]], edgecolor='none', alpha=alpha)
leg2 = ax2.legend(fontsize=legend_size, loc='lower right', ncol=2)

ax2.set_xlim(0, obs_window)
ax2.set_ylim(-50, 65)
ax2.tick_params(axis='x', labelsize=tick_size)
ax2.tick_params(axis='y', labelsize=tick_size)
ax2.set_xlabel('Evaluation epochs', fontsize=label_size)
ax2.set_ylabel('Evaluation reward', fontsize=label_size)
ax2.ticklabel_format(axis="x")

ax2.grid()
# set the linewidth of each legend object
for legobj in leg2.legendHandles:
    legobj.set_linewidth(2.0)

plt.tight_layout()
plt.savefig("benchmark_training.pdf")
plt.show()

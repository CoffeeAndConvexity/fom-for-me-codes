import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# parser = argparse.ArgumentParser()
# parser.add_argument('--n', type=int, default=50, help='n = number of buyers')
# parser.add_argument('--m', type=int, default=100, help='m = number of items')
# args = parser.parse_args()
# n, m = args.n, args.m

n, m = 50, 100 # size of the market
sd = 3 # the sample path to plot

sns.set() # grid background in plots

save_figure = False
res_measure = 'dgap' # p_diff or dgap

column_names = ['sd', 'n', 'm', 'distr', 'algo', 'iter', 'p_diff', 'dgap', 'ls']

fpath = "logs/{}.csv".format(sd)
df_list = [pd.read_csv(fpath, names = column_names).astype({'sd':int, 'n': int, 'm': int, 'distr': str, 'algo': str, 'iter': int, 'p_diff': float, 'ls': float}) for fpath in fpaths] # ls is int but we have missing values here

# all seeds, all m, n, all distr, pg-ls and pr
logs_all = pd.concat(df_list, axis=0, ignore_index=True)

all_seeds = logs_all['sd'].unique() # all seed values

logs_pgls = logs_all[logs_all['algo'] == 'pgls']
logs_pr = logs_all[logs_all['algo'] == 'pr']

# average across seeds
logs_pgls_mean = logs_pgls.groupby(['n', 'm', 'iter', 'distr'], as_index = False).mean()
logs_pr_mean = logs_pr.groupby(['n', 'm', 'iter', 'distr'], as_index = False).mean()

# plot: for pr, x-axis is iter; for pgls, x-axis is ave. ls. count
distr_list = ['unif', 'normal', 'exp', 'log_normal']
fig, ax = plt.subplots(1, 4, sharex = True, sharey = True, constrained_layout = True, figsize=(12, 3))
distr_titles = ['Uniform', 'Normal', 'Exponential', 'Lognormal']
for idx in range(4):
    distr = distr_list[idx]
    ax[idx].plot('ls', res_measure, data = logs_pgls_mean[logs_pgls_mean['distr'] == distr], label='PGLS', color = 'orange')
    ax[idx].plot('iter', res_measure, data = logs_pr_mean[logs_pr_mean['distr'] == distr], label='PR', color = 'blue')
    ax[idx].set_title(distr_titles[idx])
    
# log scale for all
[(a.set_xscale('log'), a.set_yscale('log')) for a in ax]
# legends
handles, labels = ax[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles)) # do not duplicate labels in legend
fig.legend(by_label.values(), by_label.keys(), loc='upper right', ncol=1, bbox_to_anchor=(0.65, 0.5))
# axis labels
fig.text(0.5, -0.05, r'Iteration $t$', ha='center')
fig.text(-0.03, 0.5, r'duality gap', va='center', rotation='vertical')
# title
fig.suptitle("n = {}, m = {}".format(n, m))

if save_figure == True:
    plt.savefig("plots/pgls-vs-pr-n-{}-m-{}-sd-{}-dgap.pdf".format(n, m, sd), bbox_inches = 'tight')
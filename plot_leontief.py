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

# n, m = 50, 100 # size of the market
# sd = 3 # the sample path to plot

sns.set() # grid background in plots

save_figure = True

# the 'ls' column is for pgls only (total num. of line search steps up to curr. iter.)
column_names = ['sd', 'n', 'm', 'distr', 'algo', 'iter', 'ave_ub', 'ls']

# read all csv

varying_B = True

folder = 'logs/leontief_varying_B' if varying_B else 'logs/leontief_B_1'

fpaths = [os.path.join(folder, ss) for ss in os.listdir(folder) if ss.endswith('.csv')] 
df_list = [pd.read_csv(fpath, names = column_names).astype({'sd':int, 'n': int, 'm': int, 'distr': str, 'algo': str, 'iter': int, 'ave_ub': float, 'ls': float}) for fpath in fpaths] # ls is int but we have missing values here

# all seeds, all m, n, all distr, pg-ls and pr
logs_all = pd.concat(df_list, axis=0, ignore_index=True)
# transform to the new error measure dgap/n
# logs_all['ave_ub'] = (logs_all['ave_ub'] * logs_all['n'])**2/(2 * logs_all['n'])
# truncate right before desireed accu.

##################################################
# get accu
accu = 5e-6

logs_all = logs_all[logs_all['ave_ub'] >= accu]

# experiment setup primitives
all_seeds = logs_all['sd'].unique() # all seed values
distr_list = ['unif', 'normal', 'exp', 'log_normal']
distr_titles = ['Uniform', 'Normal', 'Exponential', 'Lognormal']

n_list = logs_all['n'].unique()
print("n_list: {}".format(n_list))

# max. iter (ls) of algos for all seeds
logs_pgls_max_ls_all_seeds = logs_all[logs_all['algo'] =='pg'].groupby(['sd', 'n', 'm', 'distr'], as_index = False)['ls'].max()
# logs_pr_max_iter_all_seeds = logs_all[logs_all['algo'] =='pr-ql-shmyrev'].groupby(['sd', 'n', 'm', 'distr'], as_index = False)['iter'].max()
# logs_fw_max_iter_all_seeds = logs_all[logs_all['algo'] =='fwls-ql-shmyrev'].groupby(['sd', 'n', 'm', 'distr'], as_index = False)['iter'].max()
# ave. over seeds
logs_pgls_max_ls_ave = logs_pgls_max_ls_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['ls'].mean()
# logs_pr_max_iter_ave = logs_pr_max_iter_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['iter'].mean()
# logs_fw_max_iter_ave = logs_fw_max_iter_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['iter'].mean()
# sd. dev. over seeds (there is a BUG in the groupby.std function)
logs_pgls_max_ls_sdev = logs_pgls_max_ls_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['ls'].var()
logs_pgls_max_ls_sdev['ls'] = logs_pgls_max_ls_sdev['ls'] ** (1/2)
# logs_pr_max_iter_sdev = logs_pr_max_iter_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['iter'].var()
# logs_pr_max_iter_sdev['iter'] = logs_pr_max_iter_sdev['iter'] ** (1/2)
# logs_fw_max_iter_sdev = logs_fw_max_iter_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['iter'].var()
# logs_fw_max_iter_sdev['iter'] = logs_fw_max_iter_sdev['iter'] ** (1/2)

# plot: for pr, x-axis is iter; for pgls, x-axis is ave. ls. count
fig, ax = plt.subplots(1, 4, sharex = True, sharey = False, constrained_layout = True, figsize=(12, 3.4))
for idx in range(4): # subplots according to sampling distribution
    distr = distr_list[idx]
    # with errorbars
    pgls_yerr = (1.96)/np.sqrt(len(all_seeds)) * logs_pgls_max_ls_sdev[logs_pgls_max_ls_sdev['distr'] == distr]['ls']
    # pr_yerr = (1.96)/np.sqrt(len(all_seeds)) * logs_pr_max_iter_sdev[logs_pr_max_iter_sdev['distr'] == distr]['iter']
    # fw_yerr = (1.96)/np.sqrt(len(all_seeds)) * logs_fw_max_iter_sdev[logs_fw_max_iter_sdev['distr'] == distr]['iter']
    ax[idx].errorbar('n', 'ls', data = logs_pgls_max_ls_ave[logs_pgls_max_ls_ave['distr'] == distr], yerr = pgls_yerr, label = 'PGLS', color = 'orange')
    # ax[idx].errorbar('n', 'iter', data = logs_pr_max_iter_ave[logs_pr_max_iter_ave['distr'] == distr], yerr = pr_yerr, label = 'PR', color = 'blue')
    # ax[idx].errorbar('n', 'iter', data = logs_fw_max_iter_ave[logs_fw_max_iter_ave['distr'] == distr], yerr = fw_yerr, label = 'FW', color = 'green')
    ax[idx].set_title(distr_titles[idx], fontsize=16)
    ax[idx].tick_params(axis='x', labelsize = 16)
    ax[idx].tick_params(axis='y', labelsize = 16)

# legends
# handles, labels = ax[0].get_legend_handles_labels()
# by_label = dict(zip(labels, handles)) # do not duplicate labels in legend
# fig.legend(by_label.values(), by_label.keys(), loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0.78), fontsize=14)
# axis labels
# fig.text(0.5, -0.05, r'$n$ ($m=2n$)', ha='center')
# fig.text(-0.03, 0.5, r'dgap$/n \leq${:.0e}'.format(accu), va='center', rotation='vertical', fontsize=18)
# title
fig.suptitle(r"Leontief utilities, dgap$/n \leq${:.0e}".format(accu), fontsize=20) if varying_B else fig.suptitle(r"Leontief utilities ($B_i=1$), dgap$/n \leq${:.0e}".format(accu), fontsize=20)

if save_figure == True:
    if varying_B:
        plt.savefig("plots/leontief_varying_B/iters-to-normalized-dgap-{:.0e}.pdf".format(accu), bbox_inches = 'tight')
    else:
        plt.savefig("plots/leontief_B_1/iters-to-normalized-dgap-{:.0e}.pdf".format(accu), bbox_inches = 'tight')
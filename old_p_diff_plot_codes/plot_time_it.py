import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

accu = 0.001

sns.set() # grid background in plots

save_figure = True

column_names = ['sd', 'cp', 'pg', 'pr', 'fw']

# read all csv files
fpaths = ["logs/linear_time_it/{}".format(ss) for ss in os.listdir('logs/linear_time_it') if ss.endswith('.csv')] 
fpaths = [xx for xx in fpaths if xx.endswith("1000.csv")]
df_list = [pd.read_csv(fpath, names = column_names).astype({'sd':int, 'cp': float, 'pg': float, 'pr': float, 'fw': float}) for fpath in fpaths] # ls is int but we have missing values here

# all seeds, all m, n, all distr, pg-ls and pr
logs_all = pd.concat(df_list, axis=0, ignore_index=True)
mean_time = logs_all.mean()
se_time = logs_all.std()/np.sqrt(10)

logs_all = logs_all[logs_all['p_diff'] >= accu]

all_seeds = logs_all['sd'].unique() # all seed values
distr_list = ['unif', 'normal', 'exp', 'log_normal']
distr_titles = ['Uniform', 'Normal', 'Exponential', 'Lognormal']
# sizes = [(25, 50), (50, 100), (75, 150), (100, 200)]
# n_list = [25, 50, 75, 100]
n_list = logs_all['n'].unique()
print("n_list: {}".format(n_list))

# max. iter (ls) of algos for all seeds
logs_pgls_max_ls_all_seeds = logs_all[logs_all['algo'] =='pgls'].groupby(['sd', 'n', 'm', 'distr'], as_index = False)['ls'].max()
logs_pr_max_iter_all_seeds = logs_all[logs_all['algo'] =='pr'].groupby(['sd', 'n', 'm', 'distr'], as_index = False)['iter'].max()
logs_fw_max_iter_all_seeds = logs_all[logs_all['algo'] =='fwls'].groupby(['sd', 'n', 'm', 'distr'], as_index = False)['iter'].max()
# ave. over seeds
logs_pgls_max_ls_ave = logs_pgls_max_ls_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['ls'].mean()
logs_pr_max_iter_ave = logs_pr_max_iter_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['iter'].mean()
logs_fw_max_iter_ave = logs_fw_max_iter_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['iter'].mean()
# sd. dev. over seeds (there is a BUG in the .groupby.std function)
logs_pgls_max_ls_sdev = logs_pgls_max_ls_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['ls'].var()
logs_pgls_max_ls_sdev['ls'] = logs_pgls_max_ls_sdev['ls'] ** (1/2)
logs_pr_max_iter_sdev = logs_pr_max_iter_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['iter'].var()
logs_pr_max_iter_sdev['iter'] = logs_pr_max_iter_sdev['iter'] ** (1/2)
logs_fw_max_iter_sdev = logs_fw_max_iter_all_seeds.groupby(['n', 'm', 'distr'], as_index = False)['iter'].var()
logs_fw_max_iter_sdev['iter'] = logs_fw_max_iter_sdev['iter'] ** (1/2)

# plot: for pr, x-axis is iter; for pgls, x-axis is ave. ls. count
fig, ax = plt.subplots(1, 4, sharex = True, sharey = False, constrained_layout = True, figsize=(12, 3))
for idx in range(4): # subplots according to sampling distribution
    distr = distr_list[idx]
    # with errorbars
    pgls_yerr = (1.96)/np.sqrt(len(all_seeds)) * logs_pgls_max_ls_sdev[logs_pgls_max_ls_sdev['distr'] == distr]['ls']
    pr_yerr = (1.96)/np.sqrt(len(all_seeds)) * logs_pr_max_iter_sdev[logs_pr_max_iter_sdev['distr'] == distr]['iter']
    fw_yerr = (1.96)/np.sqrt(len(all_seeds)) * logs_fw_max_iter_sdev[logs_pr_max_iter_sdev['distr'] == distr]['iter']
    ax[idx].errorbar('n', 'ls', data = logs_pgls_max_ls_ave[logs_pgls_max_ls_ave['distr'] == distr], yerr = pgls_yerr, label = 'PGLS', color = 'orange')
    ax[idx].errorbar('n', 'iter', data = logs_pr_max_iter_ave[logs_pr_max_iter_ave['distr'] == distr], yerr = pr_yerr, label = 'PR', color = 'blue')
    ax[idx].errorbar('n', 'iter', data = logs_fw_max_iter_ave[logs_pr_max_iter_ave['distr'] == distr], yerr = fw_yerr, label = 'FW', color = 'green')
    ax[idx].set_title(distr_titles[idx])

# legends
handles, labels = ax[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles)) # do not duplicate labels in legend
fig.legend(by_label.values(), by_label.keys(), loc='upper right', ncol=1, bbox_to_anchor=(0.68, 0.9))
# axis labels
fig.text(0.5, -0.05, r'$n$ ($m=2n$)', ha='center')
fig.text(-0.03, 0.5, r'num. iter. to $\epsilon(p,p^*) \leq {}$'.format(accu), va='center', rotation='vertical')
############## title (one of the following) ##############
# fig.suptitle(r"Linear utilities ($B_i = 1$)")
fig.suptitle(r"Linear utilities ($B_i = 1$)")

# if save_figure == True:
#     plt.savefig("plots/linear/iters-to-p_diff-{:.0e}.pdf".format(accu), bbox_inches = 'tight')
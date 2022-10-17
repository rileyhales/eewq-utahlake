import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

LOG = True

path = '_nolog'
if LOG:
    path = '_log'


metrics_table = pd.read_csv(f'results{path}/kfolds_alpha_metrics_table.csv')

fig, ax1 = plt.subplots(tight_layout=True, figsize=(10, 5), dpi=1800)
fig.suptitle(f'LASSO Model RMSE and Term Count vs L1 Alpha Weight{" (LOG)" if LOG else ""}', fontsize=14, fontweight='bold')

ax2 = plt.twinx(ax1)
ax1.minorticks_on()
ax2.minorticks_on()
ax1.grid(True, which='both', axis='both', linestyle='dotted', linewidth=0.5)
ax1.set_ylabel('Train & Test RMSE')
ax1.set_xlabel('L1 Alpha Parameter')
ax2.set_ylabel('Term Count')

ax1.plot(metrics_table['alpha'], np.sqrt(metrics_table['mse_test_mean']), '-o', markersize=2, color='red', label='Test RMSE (KFolds Mean)', linewidth=1)
ax1.fill_between(metrics_table['alpha'], np.sqrt(metrics_table['mse_test_75']), np.sqrt(metrics_table['mse_test_25']), color='red', alpha=0.2, label='Test RMSE (KFolds 25-75%)')

ax1.plot(metrics_table['alpha'], np.sqrt(metrics_table['mse_train_mean']), '-o', markersize=2, color='green', label='Train RMSE (KFolds Mean)', linewidth=1)
ax1.fill_between(metrics_table['alpha'], np.sqrt(metrics_table['mse_train_75']), np.sqrt(metrics_table['mse_train_25']), color='green', alpha=0.2, label='Train RMSE (KFolds 25-75%)')

ax2.plot(metrics_table['alpha'], metrics_table['term_count_mode'], color='blue', label='Term Count (Mode)')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2)

fig.savefig(f'figures{path}/error_vs_term_count.png')

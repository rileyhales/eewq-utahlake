import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

metrics_table = pd.read_csv('results/kfolds_alpha_metrics_table.csv')
alpha_table = pd.read_csv('results/alpha_varied_iteration_metrics2.csv')

fig, ax1 = plt.subplots(tight_layout=True, figsize=(8, 6), dpi=700)
fig.suptitle('LASSO Model RMSE and Term Count vs L1 Alpha Weight', fontsize=14, fontweight='bold')

ax2 = plt.twinx(ax1)
ax1.minorticks_on()
ax2.minorticks_on()
ax1.grid(True, which='both', axis='both', linestyle='dotted', linewidth=0.5)
ax1.set_ylabel('Test RMSE')
ax1.set_xlabel('L1 Alpha Parameter')
ax2.set_ylabel('Term Count')

metrics_table = metrics_table[metrics_table['alpha'] <= 100]

test_fit = np.polyval(np.polyfit(metrics_table['alpha'], metrics_table['rmse_test_median'], 2), metrics_table['alpha'])
ax1.plot(metrics_table['alpha'], metrics_table['rmse_test_median'], '-o', markersize=2, color='red', label='Test RMSE (KFolds Median)', linewidth=0)
ax1.plot(metrics_table['alpha'], test_fit, color='red', label='Test RMSE (X^2 Fit)', linewidth=1)

train_fit = np.polyval(np.polyfit(metrics_table['alpha'], metrics_table['rmse_train_median'], 2), metrics_table['alpha'])
ax1.plot(metrics_table['alpha'], metrics_table['rmse_train_median'], '-o', markersize=2, color='green', label='Train RMSE (KFolds Median)', linewidth=0)
ax1.plot(metrics_table['alpha'], train_fit, color='green', label='Train RMSE (X^2 Fit)', linewidth=1)

# ax2.plot(metrics_table['alpha'], metrics_table['AIC'], '-o', color='purple', label='AIC (axis 2)', linewidth=1)
ax2.plot(metrics_table['alpha'], metrics_table['term_count_mode'], color='blue', label='Term Count (Mode)')
# ax2.plot(metrics_table['alpha'], metrics_table['term_count_min'], color='green', label='Term Count (Min)')
# ax2.plot(metrics_table['alpha'], metrics_table['term_count_max'], color='green', label='Term Count (Max)')

fig.legend(loc='upper center', bbox_to_anchor=(.5, 0.9), ncol=1)

plt.show()

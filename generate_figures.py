import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

metrics_table = pd.read_csv('results/kfolds_alpha_metrics_table.csv')
alpha_table = pd.read_csv('results/alpha_varied_iteration_metrics2.csv')

fig, ax1 = plt.subplots(tight_layout=True, figsize=(8, 6), dpi=500)
ax2 = plt.twinx(ax1)

ax1.minorticks_on()

ax1.plot(metrics_table['alpha'], metrics_table['rmse_test_mean'], '-o', markersize=6, color='red', label='Test RMSE')
ax2.plot(metrics_table['alpha'], metrics_table['term_count_mode'], '-o', markersize=6, color='blue', label='Term Count')

# ax1.plot(alpha_table['alpha'], np.sqrt(alpha_table['mse_test']), '-o', markersize=6, color='red', label='Test RMSE')
# ax2.plot(alpha_table['alpha'], alpha_table['term_count'], '-o', markersize=6, color='blue', label='Term Count')
ax1.set_ylabel('Test RMSE')
ax1.set_xlabel('L1 Alpha Parameter')
ax2.set_ylabel('Term Count')

# fig.legend(loc='upper center', ncol=2)
fig.legend(loc='center right', bbox_to_anchor=(.88, 0.5), ncol=1)
fig.suptitle('LASSO Model RMSE and Term Count vs L1 Alpha Weight', fontsize=14, fontweight='bold')


plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

df = pd.read_csv('results_nolog/kfolds_alpha_coefficients_table.csv', index_col=0)
del df['alpha']
df = df.reindex(df.sum().sort_values(ascending=False).index, axis=1)
df.index.name = 'Feature Count (Mode)'
# convert to percentages
df = df / np.nanmax(df.values)

# Draw a heatmap with the numeric values in each cell
fig, ax = plt.subplots(figsize=(15, 10), dpi=2000, tight_layout=True)
sns.heatmap(df, cmap=plt.cm.Blues, annot=True, fmt=".1%", linewidths=.5, ax=ax, cbar=False)
plt.setp(ax.get_xticklabels(), rotation=-40, ha='center', va='top')
ax.set_title('Term Count vs Feature Selection Frequency', fontsize=14, fontweight='bold')
fig.savefig('figures/feature_heatmap_seaborn.png')
fig.show()

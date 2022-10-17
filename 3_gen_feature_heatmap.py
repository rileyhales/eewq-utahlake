import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LOG = True

path = '_nolog'
if LOG:
    path = '_log'

sns.set_theme()

df = pd.read_csv(f'results{path}/kfolds_alpha_coefficients_table.csv')
del df['alpha']
df = df.reindex(df.sum().sort_values(ascending=False).index, axis=1)
df.index.name = 'Feature Count (Mode)'
# convert to percentages
df = df / np.nanmax(df.values)

# Draw a heatmap with the numeric values in each cell
fig, ax = plt.subplots(figsize=(15, 10), dpi=1800, tight_layout=True)
sns.heatmap(df, cmap=plt.cm.Blues, annot=True, fmt=".1%", linewidths=.5, ax=ax, cbar=False)
plt.setp(ax.get_xticklabels(), rotation=-40, ha='center', va='top')
ax.set_title(f'Term Count vs Feature Selection Frequency{" (LOG)" if LOG else ""}', fontsize=14, fontweight='bold')
fig.savefig(f'figures{path}/feature_heatmap_seaborn.png')


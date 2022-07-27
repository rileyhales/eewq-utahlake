import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('results/kfolds_alpha_coefficients_table.csv', index_col=0)
del df['alpha']
df = df.reindex(df.sum().sort_values(ascending=False).index, axis=1)

x_labels = df.columns
y_labels = df.index
values = df.values / np.nanmax(df.values) * 100

fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True, dpi=700)
ax.set_title("Frequency of Feature Selection vs Term Count", fontsize=14, fontweight='bold')
ax.imshow(values, cmap=plt.cm.Blues)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
ax.set_xlabel('Feature Names')
ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
ax.set_ylabel('Term Count (Mode)')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations
values = np.round(values).astype(int)
for i in range(len(y_labels)):
    for j in range(len(x_labels)):
        label = f'{values[i, j]}%' if values[i, j] > 0 else ""
        text = ax.text(j, i, label, ha="center", va="center")

plt.show()
fig.savefig('feature_heatmap.png')

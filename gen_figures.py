import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

metrics_df = pd.read_csv('results/iteration_metrics.csv')
coef_df = pd.read_csv('results/iteration_coefficients.csv')

coef_df[coef_df == 0] = np.nan
coef_df = coef_df.dropna(axis=1, how='all')
coef_df.describe().to_csv('coef_describe.csv')

pd.plotting.scatter_matrix(coef_df, figsize=(15, 15))  # .tofile('coef_scatter_matrix.png')
plt.tight_layout()
plt.show()
coef_df.hist(figsize=(13, 13))
plt.tight_layout()
plt.show()

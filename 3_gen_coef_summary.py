import pandas as pd
import glob


for csv in glob.glob('results/kfolds_alpha_*_iteration_coefficients.csv'):
    print(csv)

cdf = pd.read_csv('results/alpha_varied_iteration_metrics.csv')
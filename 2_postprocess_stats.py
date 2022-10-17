import pandas as pd
import numpy as np
import glob

num_samples = pd.read_csv('modelinput_uncorrectedChla_30m_v03.csv').shape[0]

metrics = ['me', 'mse', 'rmse']
stats = ['mean', 'median', 'std', 'min', 'max', '25', '75']
train_test = ['train', 'test']
metric_labels = [f'{m}_{tt}_{stat}' for m in metrics for tt in train_test for stat in stats]

table_val = {
    "alpha": [],
    "term_count_mode": [],
    "term_count_min": [],
    "term_count_max": [],
}
for label in metric_labels:
    table_val[label] = []

for csv in sorted(glob.glob('results/kfolds_alpha_*metrics.csv')):
    alpha = float(csv.split('_')[2])

    df = pd.read_csv(csv)

    table_val["alpha"].append(alpha)

    table_val["term_count_mode"].append(df["term_count"].mode().values[0])
    table_val["term_count_min"].append(df["term_count"].min())
    table_val["term_count_max"].append(df["term_count"].max())

    for metric in [f'{m}_{tt}' for m in metrics for tt in train_test]:
        table_val[f"{metric}_min"].append(df[metric].min())
        table_val[f"{metric}_max"].append(df[metric].max())
        table_val[f"{metric}_mean"].append(df[metric].mean())
        table_val[f"{metric}_median"].append(df[metric].median())
        table_val[f"{metric}_std"].append(df[metric].std())
        table_val[f"{metric}_25"].append(df[metric].quantile(0.25))
        table_val[f"{metric}_75"].append(df[metric].quantile(0.75))

df = pd.DataFrame(table_val)
df = df.sort_values(by='alpha')
df.to_csv('results/kfolds_alpha_metrics_table.csv', index=False)


df_master = pd.DataFrame()
for csv in sorted(glob.glob('results/kfolds_alpha_*coefficients.csv')):
    try:
        df = pd.read_csv(csv)
        alpha = float(csv.split('_')[2])
        df[df == 0] = np.nan
        df = df.dropna(axis=1, how='all')
        a = df.count()
        a['alpha'] = alpha
        a['term_count'] = df['term_count'].mode()
        df_master = pd.concat([df_master, pd.DataFrame(a).transpose()])
    except Exception as e:
        print(e)
        print(csv)

df_master.groupby('term_count').median().sort_index(ascending=False).to_csv('results/kfolds_alpha_coefficients_table.csv')

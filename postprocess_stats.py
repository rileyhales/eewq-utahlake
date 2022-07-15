import pandas as pd
import numpy as np
import glob

for csv in sorted(glob.glob('results/kfolds_alpha_*coefficients.csv')):
    df = pd.read_csv(csv)
    alpha = int(csv.split('_')[2])
    df[df == 0] = np.nan
    df = df.dropna(axis=1, how='all')
    df.describe().to_csv(f'results/kfolds_alpha_{alpha}_coef_describe.csv')

table_val = {
    "alpha": [],

    "term_count_mode": [],
    "term_count_min": [],
    "term_count_max": [],

    "me_test_mean": [],
    "me_test_median": [],
    "me_test_std": [],
    "me_test_min": [],
    "me_test_max": [],
    "me_test_25": [],
    "me_test_75": [],

    "mse_test_mean": [],
    "mse_test_median": [],
    "mse_test_std": [],
    "mse_test_min": [],
    "mse_test_max": [],
    "mse_test_25": [],
    "mse_test_75": [],

    "rmse_test_mean": [],
    "rmse_test_median": [],
    "rmse_test_std": [],
    "rmse_test_min": [],
    "rmse_test_max": [],
    "rmse_test_25": [],
    "rmse_test_75": [],

    "me_train_mean": [],
    "me_train_median": [],
    "me_train_std": [],
    "me_train_min": [],
    "me_train_max": [],
    "me_train_25": [],
    "me_train_75": [],

    "mse_train_mean": [],
    "mse_train_median": [],
    "mse_train_std": [],
    "mse_train_min": [],
    "mse_train_max": [],
    "mse_train_25": [],
    "mse_train_75": [],
}

for csv in sorted(glob.glob('results/kfolds_alpha_*metrics.csv')):
    alpha = int(csv.split('_')[2])

    df = pd.read_csv(csv)
    df['rmse_test'] = np.sqrt(df['mse_test'])
    df['rmse_train'] = np.sqrt(df['mse_train'])

    table_val["alpha"].append(alpha)

    table_val["term_count_mode"].append(df["term_count"].mode().values[0])
    table_val["term_count_min"].append(df["term_count"].min())
    table_val["term_count_max"].append(df["term_count"].max())

    table_val["me_test_mean"].append(df["me_test"].mean())
    table_val["me_test_median"].append(df["me_test"].median())
    table_val["me_test_std"].append(df["me_test"].std())
    table_val["me_test_min"].append(df["me_test"].min())
    table_val["me_test_max"].append(df["me_test"].max())
    table_val["me_test_25"].append(df["me_test"].quantile(0.25))
    table_val["me_test_75"].append(df["me_test"].quantile(0.75))

    table_val["mse_test_mean"].append(df["mse_test"].mean())
    table_val["mse_test_median"].append(df["mse_test"].median())
    table_val["mse_test_std"].append(df["mse_test"].std())
    table_val["mse_test_min"].append(df["mse_test"].min())
    table_val["mse_test_max"].append(df["mse_test"].max())
    table_val["mse_test_25"].append(df["mse_test"].quantile(0.25))
    table_val["mse_test_75"].append(df["mse_test"].quantile(0.75))

    table_val["rmse_test_mean"].append(np.sqrt(df["mse_test"].sum()))
    table_val["rmse_test_median"].append(df["rmse_test"].median())
    table_val["rmse_test_std"].append(df["rmse_test"].std())
    table_val["rmse_test_min"].append(df["rmse_test"].min())
    table_val["rmse_test_max"].append(df["rmse_test"].max())
    table_val["rmse_test_25"].append(df["rmse_test"].quantile(0.25))
    table_val["rmse_test_75"].append(df["rmse_test"].quantile(0.75))

    table_val["me_train_mean"].append(df["me_train"].mean())
    table_val["me_train_median"].append(df["me_train"].median())
    table_val["me_train_std"].append(df["me_train"].std())
    table_val["me_train_min"].append(df["me_train"].min())
    table_val["me_train_max"].append(df["me_train"].max())
    table_val["me_train_25"].append(df["me_train"].quantile(0.25))
    table_val["me_train_75"].append(df["me_train"].quantile(0.75))

    table_val["mse_train_mean"].append(df["mse_train"].mean())
    table_val["mse_train_median"].append(df["mse_train"].median())
    table_val["mse_train_std"].append(df["mse_train"].std())
    table_val["mse_train_min"].append(df["mse_train"].min())
    table_val["mse_train_max"].append(df["mse_train"].max())
    table_val["mse_train_25"].append(df["mse_train"].quantile(0.25))
    table_val["mse_train_75"].append(df["mse_train"].quantile(0.75))

pd.DataFrame(table_val).sort_values(by='alpha').to_csv('results/kfolds_alpha_metrics_table.csv', index=False)

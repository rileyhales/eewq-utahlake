import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error

LOG = True

obs_val_col = 'measurement value'
prepared_data = pd.read_csv('modelinput_uncorrectedChla_30m_v03.csv')
prepared_data = prepared_data[prepared_data['time window'] <= 72]
y = prepared_data[obs_val_col].values
if LOG:
    y = np.log10(y)
prepared_data = prepared_data.drop(columns=[obs_val_col, 'time window'])

X = prepared_data.values
iter_metrics_labels = ['me_test', 'mse_test', 'rmse_test', 'me_train', 'mse_train', 'rmse_train']

for alpha in np.linspace(.1, 10, 100):
    alpha = round(alpha, 1)
    print(f'alpha: {alpha}')
    coefficients = []
    iteration_metrics = []

    for train_index, test_index in RepeatedKFold(n_splits=10, n_repeats=20).split(X):
        # split the train test data with the current kfold split indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # fit lasso
        model = Lasso(alpha, max_iter=int(1e5))
        model.fit(X_train, y_train)

        # record the coefficients used in this run
        coefficients.append(model.coef_)

        # record the test metrics for the predictions made this run
        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)

        if LOG:
            pred_test = np.power(10, pred_test)
            pred_train = np.power(10, pred_train)
            y_test = np.power(10, y_test)
            y_train = np.power(10, y_train)

        test_mse = mean_squared_error(y_test, pred_test)
        test_rmse = np.sqrt(test_mse)
        test_me = np.mean(y_test - pred_test)

        train_mse = mean_squared_error(y_train, pred_train)
        train_rmse = np.sqrt(train_mse)
        train_me = np.mean(y_train - pred_train)

        iteration_metrics.append([test_me, test_mse, test_rmse, train_me, train_mse, train_rmse])

    coef_df = pd.DataFrame(coefficients, columns=prepared_data.columns)
    term_counts = coef_df.apply(lambda x: len(x[x != 0]), axis=1)
    coef_df['alpha'] = alpha
    coef_df['term_count'] = term_counts
    coef_df.to_csv(f'results/kfolds_alpha_{alpha}_iteration_coefficients.csv', index=False)

    metr_df = pd.DataFrame(iteration_metrics, columns=iter_metrics_labels)
    metr_df['alpha'] = alpha
    metr_df['term_count'] = term_counts
    metr_df.to_csv(f'results/kfolds_alpha_{alpha}_iteration_metrics.csv', index=False)

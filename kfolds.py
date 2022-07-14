import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error

prepared_data = pd.read_csv('prepared_data.csv')
X = prepared_data.drop(columns=['Result Value', ]).values
y = prepared_data['Result Value'].values

for alpha in [1, 5, 10, 15]:
    coefficients = []
    iteration_metrics = []
    iter_metrics_labels = ['me_test', 'mse_test', 'me_train', 'mse_train']

    rkfolder = RepeatedKFold(n_splits=20, n_repeats=10)
    for i, (train_index, test_index) in enumerate(rkfolder.split(X)):
        print(i)
        # split the train test data with the current kfold split indices
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit lasso
        model = Lasso(alpha, max_iter=int(1e9))
        model.fit(X_train, y_train)

        # record the coefficients used in this run
        coefficients.append(model.coef_)

        # record the test metrics for the predictions made this run
        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)
        test_mse = mean_squared_error(y_test, pred_test)
        test_me = np.mean(y_test - pred_test)
        train_mse = mean_squared_error(y_train, pred_train)
        train_me = np.mean(y_train - pred_train)
        iteration_metrics.append([test_me, test_mse, train_me, train_mse])

    pd.DataFrame(coefficients, columns=prepared_data.columns[1:]).to_csv(
        f'results/kfolds_alpha_{alpha}_iteration_coefficients.csv', index=False)
    pd.DataFrame(iteration_metrics, columns=iter_metrics_labels).to_csv(
        f'results/kfolds_alpha_{alpha}_iteration_metrics.csv', index=False)

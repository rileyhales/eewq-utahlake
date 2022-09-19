import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


prepared_data = pd.read_csv('prepared_data.csv')
result_col = 'Result Value'
X = prepared_data.drop(columns=[result_col, ]).values
y = prepared_data[result_col].values

coefficients = []
iteration_metrics = []
iter_metrics_labels = ['me_test', 'mse_test', 'me_train', 'mse_train']

alpha = 100000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
model = Lasso(alpha, max_iter=int(1e9))
model.fit(X_train, y_train)

# record the coefficients used in this run
coefficients.append(model.coef_)

# record the test metrics for the predictions made this run
predictions = model.predict(X_train)
train_mse = mean_squared_error(y_train, predictions)
train_me = np.mean(y_train - predictions)
predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, predictions)
test_me = np.mean(y_test - predictions)
iteration_metrics.append([test_me, test_mse, train_me, train_mse])

coef_df = pd.DataFrame(coefficients, columns=prepared_data.columns[1:])
coef_df['alpha'] = alpha
term_counts = coef_df.apply(lambda x: len(x[x != 0]), axis=1)
coef_df['term_count'] = term_counts
coef_df.to_csv('results/alpha_varied_iteration_coefficients.csv', index=False)

metr_df = pd.DataFrame(iteration_metrics, columns=iter_metrics_labels)
metr_df['alpha'] = alpha
metr_df['term_count'] = term_counts
metr_df.to_csv('results/alpha_varied_iteration_metrics2.csv', index=False)

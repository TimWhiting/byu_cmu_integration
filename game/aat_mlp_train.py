import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

data_dir = './training_data/'

with open(data_dir + 'training_data.pickle', 'rb') as f:
    training_data = np.array(pickle.load(f))

x = training_data[:, 0:-1]
y = training_data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# param_grid = {'hidden_layer_sizes': [(4, 16, 16), (8, 16, 16), (16, 16, 16), (4, 32, 32), (8, 32, 32), (16, 32, 32), (32, 32, 32)],
#               'activation': ['relu', 'tanh', 'logistic'],
#               'max_iter': [10000]}

param_grid = {'hidden_layer_sizes': [(32, 32, 32)],
              'activation': ['tanh'],
              'max_iter': [10000]}

model = GridSearchCV(MLPRegressor(), param_grid)
model.fit(x_train_scaled, y_train)

print('Best score: ' + str(model.best_score_))
print('Best parameters: ' + str(model.best_params_))

results = (y_test - model.predict(x_test_scaled))**2
print(np.mean(results), np.max(results), np.min(results), np.var(results))

with open(data_dir + 'trained_mlp_aat.pickle', 'wb') as f:
    pickle.dump(model, f)

with open(data_dir + 'trained_mlp_scaler_aat.pickle', 'wb') as f:
    pickle.dump(scaler, f)

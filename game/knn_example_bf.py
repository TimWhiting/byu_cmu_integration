import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import random
import numpy as np

k = 14
n_samples = 15000

data_dir = './training_data/'

with open(data_dir + 'training_data_najma.pickle', 'rb') as f:
	training_data = pickle.load(f)

print(len(training_data))

# randomly select samples
if n_samples < len(training_data):
	sample_idxs = random.sample(list(range(len(training_data))), n_samples)
	training_data_filtered = [training_data[sample_idx] for sample_idx in sample_idxs]
else:
	training_data_filtered = training_data.copy()

# format the data into inputs and outputs
X = [example[0:-1] for example in training_data_filtered]
Y = [example[-1] for example in training_data_filtered]

train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.3)

# fit a model
parameters = {'weights':('uniform', 'distance'), 'n_neighbors':list(range(1,30))}
template_model = KNeighborsRegressor()
model = GridSearchCV(template_model, parameters, scoring="neg_mean_squared_error")
model.fit(train_x,train_y)

results = (test_y - model.predict(test_x))**2
print(np.mean(results), np.max(results), np.min(results), np.var(results))

with open("trained_chief_estimation_model_najma", 'wb') as f:
#with open("trained_chief_estimation_model", 'wb') as f:
	pickle.dump(model, f)
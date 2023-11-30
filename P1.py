import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("drug.csv")

# Check whether there are missing values
missing_values = data.isnull().sum()
print('Missing values:\n', missing_values)

print('----------------------')

# Check the type of each feature (categorical or numerical)
data_types = data.dtypes
print('Data types:\n', data_types)

print('----------------------')

# Records containing missing values are removed
if data.isnull().values.any():
    data_cleaned_rows = data.dropna()
else:
    data_cleaned_rows = data.copy()

data_types = data_cleaned_rows.dtypes

label_encoder = LabelEncoder()
count = 0

# Categorical features are encoded
for column_name in data_cleaned_rows.columns:
    if data_types[column_name] == 'object':
        data_cleaned_rows.loc[:, column_name] = label_encoder.fit_transform(data_cleaned_rows[column_name])


# The features and targets are separated
x = data_cleaned_rows.drop(columns=['Drug'])
y = data_cleaned_rows[['Drug']]

#DecisionTreeRegressor
# Number of experiments
num_experiments = 5

# Initialize lists to store mae and size of trees
mae_values = []
tree_sizes = []

for i in range(num_experiments):
    # Split the data into training and testing sets with different random states
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)

    # Convert data to Numpy array
    x_train = x_train.to_numpy().reshape((-1, 5))
    x_test = x_test.to_numpy().reshape((-1, 5))
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()

    # train the Decision Tree
    dt_regressor = DecisionTreeRegressor()
    dt_regressor.fit(x_train, y_train)

    # Get the size of the decision tree
    tree_size = dt_regressor.tree_.node_count
    tree_sizes.append(tree_size)
    print(f"Experiment {i + 1} - Decision Tree Size: {tree_size}")

    # Make predictions
    y_pred = dt_regressor.predict(x_test)

    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    mae_values.append(mae)
    print(f"Experiment {i + 1} - Mean Absolute Error: {mae:.4f}")

# Find the index of the experiment with the lowest mean absolute error
best_experiment_index = np.argmin(mae_values)
best_mae = mae_values[best_experiment_index]
best_tree_size = tree_sizes[best_experiment_index]

# Print results
print(f"\nBest Model (Experiment {best_experiment_index + 1}):")
print(f"Lowest Mean Absolute Error: {best_mae:.4f}")
print(f"Decision Tree Size: {best_tree_size}")

best_model = DecisionTreeRegressor()
best_model.fit(x, y)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("drug.csv")

# Check whether there are missing values
missing_values = data.isnull().sum()
print('Missing values:\n', missing_values)

print('----------------------')

# Check the type of each feature (categorical or numerical)
data_types = data.dtypes
print('Data types:\n', data_types)

print('----------------------')
print('First experiment:')

# Records containing missing values are removed
data.loc[:, ["Age", "Na_to_K"]] = data[["Age", "Na_to_K"]].fillna(data[["Age", "Na_to_K"]].mean())


# For categorical columns
for column_name in data.columns:
    if data[column_name].dtype == 'object':
        data[column_name].fillna(data[column_name].mode()[0], inplace=True)

label_encoder = LabelEncoder()

# categorical features and targets are encoded
for column_name in data.columns:
    if data[column_name].dtype == 'object':
        data.loc[:, column_name] = label_encoder.fit_transform(data.loc[:, column_name])

x = data.drop(columns=['Drug'])
y = data['Drug']  # Update to use a Series for the target

y=label_encoder.fit_transform(y)
# Number of experiments
num_experiments = 5

# Initialize lists to store mae and size of trees
accuracy_values = []
tree_sizes = []

for i in range(num_experiments):
    # Split the data into training and testing sets with different random states
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # train the Decision Tree
    model = tree.DecisionTreeClassifier(criterion="gini")
    model.fit(x_train, y_train)

    # Get the size of the decision tree
    tree_size = model.tree_.node_count
    tree_sizes.append(tree_size)
    print(f"Experiment {i + 1} - Decision Tree Size: {tree_size}")

    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate Mean Absolute Error
    accuracy = model.score(x_test, y_test)
    accuracy_values.append(accuracy)
    print(f"Experiment {i + 1} - accuracy: {accuracy:.4f}")

# Find the index of the experiment with the Max accuracy
best_experiment_index = np.argmax(accuracy_values)
best_accuracy = accuracy_values[best_experiment_index]
best_tree_size = tree_sizes[best_experiment_index]

# Print results
print(f"\nBest Model (Experiment {best_experiment_index + 1}):")
print(f"Max accuracy: {best_accuracy:.4f}")
print(f"Decision Tree Size: {best_tree_size}")

print('----------------------')
print('Second experiment:')


mean_accuracies = []
mean_trees_size = []
split_ratios = np.arange(0.3, 0.8, 0.1)
columns = ['split_ratio','mean_accuracy', 'max_accuracy', 'min_accuracy','mean_tree_size', 'max_tree_size', 'min_tree_size']
report_df = pd.DataFrame(columns=columns)

for split_ratio in split_ratios:
    # Initialize lists to store results for each ratio
    accuracy_values = []
    tree_sizes = []

    for i in range(num_experiments):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - split_ratio)
        model = tree.DecisionTreeClassifier(criterion="gini")
        model.fit(x_train, y_train)

        tree_size = model.tree_.node_count
        tree_sizes.append(tree_size)

        accuracy = model.score(x_test, y_test)
        accuracy_values.append(accuracy)
    mean_accuracy=np.mean(accuracy_values)
    max_accuracy=max(accuracy_values)
    min_accuracy = min(accuracy_values)
    mean_tree_size = np.mean(tree_sizes)
    max_tree_size = max(tree_sizes)
    min_tree_size = min(tree_sizes)

    mean_accuracies.append(mean_accuracy)
    mean_trees_size.append(mean_tree_size)

    record = pd.DataFrame({
        'split_ratio': [split_ratio],
        'mean_accuracy':[mean_accuracy],
        'max_accuracy':[max_accuracy],
        'min_accuracy':[min_accuracy],
        'mean_tree_size':[mean_tree_size],
        'max_tree_size':[max_tree_size],
        'min_tree_size':[min_tree_size]
    })
    if report_df.empty:
        report_df = record.copy()
    else:
        report_df = pd.concat([report_df, record], ignore_index=True)



print(report_df)

# Create two plots
plt.figure(figsize=(12, 5))

# Plot accuracy against set size
plt.subplot(1, 2, 1)
plt.plot(split_ratios, mean_accuracies, label='Mean Accuracy')
plt.title('Mean Accuracy & Training Split Ratio')
plt.xlabel('Training Split Ratio')
plt.ylabel('Mean Accuracy')
plt.legend()


# Plot number of nodes in the final tree against set size
plt.subplot(1, 2, 2)
plt.plot(split_ratios, mean_trees_size, label='Mean tree size')
plt.title('Mean of tree size & Training Split Ratio')
plt.xlabel('Training Split Ratio')
plt.ylabel('Mean of tree size')
plt.legend()

plt.tight_layout()
plt.show()
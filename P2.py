import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Load the "diabetes.csv" dataset.
data = pd.read_csv('diabetes.csv')

#the features and targets are separated
x=data.drop(columns=['Outcome'])
y=data[['Outcome']]

#the data is shuffled and split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

#features are standardized
x_train_max=x_train.max()
x_train_min=x_train.min()
range= x_train_max-x_train_min
x_test= (x_test - x_train_min) / range
x_train = (x_train- x_train_min) / range

#Convert data to Numpy array
x_train = x_train.to_numpy().reshape((-1,8))
x_test = x_test.to_numpy().reshape((-1,8))
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


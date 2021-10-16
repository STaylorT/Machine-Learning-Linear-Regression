# Linear Regression Machine Learning Program
#
# Sources used:
# - https://towardsdatascience.com/master-machine-learning-multiple-linear-regression-from-scratch-with-python-ac716a9b78a4
# Sean Taylor Thomas

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams['figure.figsize'] = (14,7)
# rcParams['axes.spines.top'] = False
# rcParams['axes.spins.right'] = False

def import_data(filename):
    """ Take data from txt file"""
    dataset = list()
    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        dataset.append(line.split())

    return dataset


def str_column_to_float(dataset, column):
    """ Convert string column to float """
    for row in dataset:
        row[column] = float(row[column].strip())


class LinearRegression:
    """ Implementation of linear regression using gradient descent"""
    def __init__(self, l_rate = 0.7, iterations=1000):
        self.l_rate = l_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss =[]

    @staticmethod
    def _mean_squared_error(y, y_hat):
        """ Evaluating loss at each iteration
        y = array of known values
        y_hat = array of predicted values
        returns float representing error"""
        error = 0
        for i in range(len(y)):
            error += (y[i] - y_hat[i]) **2
        return error / len(y)

    def fit(self, X, y):

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for i in range(self.iterations):
            y_hat = np.dot(X, self.weights) + self.bias
            loss = self._mean_squared_error(y, y_hat)
            self.loss.append(loss)

            deriv_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (y_hat - y)))
            deriv_d = (1 / X.shape[0]) * (2 * np.sum(y_hat - y))

            self.weights -= self.l_rate * deriv_w
            self.bias -= self.l_rate * deriv_d

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


from sklearn.datasets import load_diabetes

data = load_diabetes()
x = data.data
y = data.target

filename = 'housing.data'
x = import_data(filename)

# put data in x and target (dependent var) data in y
for i in range(len(x[0])):
    str_column_to_float(x, i)
y = list()
for row in x:
    y.append(row[-1])
    row.remove(row[-1])  # separate x (independent vars) from y (dependent var)

# put into numpy arrays and normalize data
x = np.array(x)
y = np.array(y)
xnorm = np.linalg.norm(x)
x = x / xnorm

# split data into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, t_test = train_test_split(x,y, test_size = .2, random_state = 42)

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(x_test)

print(predictions)
xs = np.arange(len(model.loss))
ys = model.loss

# Plotting our loss over iterations
plt.plot(xs, ys, lw=3, c='#0033a3')
plt.title('Loss per iteration(MSE)', size=20)
plt.xlabel('Iteration', size=14)
plt.ylabel('Loss', size=14)
plt.show()

# test over different learning rates
# losses = {}
# for lr in [.7,0.5, 0.1, 0.01, 0.001]:
#     model = LinearRegression(l_rate=lr)
#     model.fit(x_train, y_train)
#     losses[f'LR={str(lr)}'] = model.loss
#
# xs = np.arange(len(model.loss))
# plt.plot(xs, losses['LR=0.7'], lw=3, label=f"LR = 0.7, Final = {losses['LR=0.7'][-1]:.2f}")
# plt.plot(xs, losses['LR=0.5'], lw=3, label=f"LR = 0.5, Final = {losses['LR=0.5'][-1]:.2f}")
# plt.plot(xs, losses['LR=0.1'], lw=3, label=f"LR = 0.1, Final = {losses['LR=0.1'][-1]:.2f}")
# plt.plot(xs, losses['LR=0.01'], lw=3, label=f"LR = 0.01, Final = {losses['LR=0.01'][-1]:.2f}")
# plt.plot(xs, losses['LR=0.001'], lw=3, label=f"LR = 0.001, Final = {losses['LR=0.001'][-1]:.2f}")
# plt.title('Loss per iteration (MSE) across l_rates', size=20)
# plt.xlabel('Iteration', size=14)
# plt.ylabel('Loss', size=14)
# plt.legend()
# plt.show()

# User predictions:
num_cols = len(x[0])
user_input = input("Would you like to provide input for prediction? y/n")
iter1 = 0
x1 = list()  # user x
while user_input == 'y' and iter1 < num_cols:
    user_x = input("Attribute %d : " % iter1)
    if not(user_x == '' or user_x == " " or user_x == "\n"):
        x1.append(float(user_x))
        iter1 += 1
if (user_input == 'y'):
    x1 = x1 / xnorm
    user_prediction = model.predict(x1)
    print(x1)
    print("Prediction : ", user_prediction)

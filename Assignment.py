import numpy as np
import matplotlib.pyplot as plt
import operator
from itertools import combinations_with_replacement

from numpy import testing

def generate_data(n):
    X = np.random.uniform(0,1,n)
    X = np.sin(2*np.pi*X)
    mu, sigma = 0, 0.3
    noise = np.random.normal(mu, sigma, n)
    y = X + noise
    return X,y

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    split_i = len(y) - int(len(y) // (1 / test_size))
    print(len(y), split_i)
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]
    return X_train, X_test, y_train, y_test

X,y = generate_data(10)
X =  np.reshape(X, (-1,1)) 
X_train, X_test, y_train, y_test = train_test_split(X,y)
print(X_train, X_test, y_train, y_test)

plt.plot(X,y,'b.')
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
_ = plt.axis([-1.5,1.5,-1.5,1.5])
plt.grid()

def mean_squared_error(y_true, y_pred):
    mse = np.mean(0.5*np.power(y_true - y_pred, 2))
    return mse

class LinearRegression(object):
    def __init__(self, n_iterations=100, learning_rate=0.001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        
    def initialize_weights(self, n_features):
        limit = 1 / np.power(n_features,0.5)
        self.w = np.random.uniform(-limit, limit, (n_features, ))
        
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            mse = mean_squared_error(y,y_pred)
            self.training_errors.append(mse)
            grad_w = -(y - y_pred).dot(X)
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

model = LinearRegression( n_iterations=3000, learning_rate=0.05)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
training_mse = model.training_errors[-1]
testing_mse = mean_squared_error(y_test, y_pred)

y_pred_line = model.predict(X)
print (training_mse,testing_mse)

m1 = plt.scatter(X_train, y_train, color='red', s=15)
m2 = plt.scatter(X_test, y_test, color='black', s=15)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_pred_line), key=sort_axis)
x, y_pred_line = zip(*sorted_zip)
plt.plot(x, y_pred_line, color='blue', linewidth=1, label="Prediction")
plt.suptitle("Linear Regression",fontsize=15)
plt.title("Training MSE: {0:.2f} Testing MSE: {1:.2f}".format(training_mse,testing_mse), fontsize=10)
plt.xlabel('X',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.legend((m1, m2), ("Training data", "Test data"), loc = 'lower right')
plt.grid()
plt.show()

def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)
    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis = 1)
    return X_new

class PolynomialRegression(object):
    def __init__(self, degree, n_iterations = 3000, learning_rate = 0.001):
        self.degree = degree
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
    
    def initialize_weights(self, n_features):
        limit = 1 / np.power(n_features, 0.5)
        self.w = np.random.uniform(-limit, limit, (n_features))
    
    def fit(self, X, y):
        X = polynomial_features(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis = 1)
        self.training_errors = []
        self.initialize_weights(n_features = X.shape[1])
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            mse = mean_squared_error(y, y_pred)
            self.training_errors.append(mse)
            grad_w = -(y - y_pred).dot(X)
            self.w -= self.learning_rate * grad_w
    
    def predict(self, X):
        X = polynomial_features(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis = 1)
        y_pred = X.dot(self.w)
        return y_pred

fig = plt.figure()
plt.figure(figsize = (16, 8), dpi = 100)
fig.subplots_adjust(hspace = 1, wspace = 1)
for i in range(1, 10):
    model = PolynomialRegression(degree = i, n_iterations = 3000, learning_rate = 0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    training_mse = model.training_errors[-1]
    testing_mse = mean_squared_error(y_test, y_pred)
    y_poly_pred = model.predict(X)
    print(training_mse, testing_mse)
    ax = fig.add_subplot(3, 3, i)
    plt.subplot(3, 3, i)
    m1 = plt.scatter(X_train, y_train, color = 'red', s = 15)
    m2 = plt.scatter(X_test, y_test, color = 'black', s = 15)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, y_poly_pred), key = sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color = 'blue', linewidth = 2, label = "Prediction")
    plt.title("Training MSE: {0:.2f} Testing MSE: {1:.2f} Degree: {2}".format(training_mse, testing_mse, i), fontsize = 10)
    plt.legend((m1, m2), ("Training data", "Test data"), loc = 'lower right')
    plt.grid()
plt.show()

X, y = generate_data(100)
X = np.reshape(X, (-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train, X_test, y_train, y_test)

fig = plt.figure()
plt.figure( figsize=(16, 8), dpi=80)
fig.subplots_adjust(hspace=1, wspace=1)
for i in range(1,10):
    model = PolynomialRegression(degree = i, n_iterations=3000, learning_rate=0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    training_mse = model.training_errors[-1]
    testing_mse = mean_squared_error(y_test, y_pred)
    y_poly_pred = model.predict(X)
    print(training_mse, testing_mse)
    ax = fig.add_subplot(3, 3, i)
    plt.subplot(3, 3, i)
    m1 = plt.scatter(X_train, y_train, color='red', s=15)
    m2 = plt.scatter(X_test, y_test, color='black', s=15)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color='blue', linewidth=2, label="Prediction")
    plt.suptitle("Polynomial Regression",fontsize=20)
    plt.title("Training MSE: {0:.2f} Testing MSE: {1:.2f} Degree: {2}".format(training_mse,testing_mse,i), fontsize=10)
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.grid()
plt.show()

X, y = generate_data(1000)
X = np.reshape(X, (-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train, X_test, y_train, y_test)

fig = plt.figure()
plt.figure(figsize = (16, 8), dpi = 90)
fig.subplots_adjust(hspace = 1, wspace = 1)
model = LinearRegression(n_iterations = 3000, learning_rate = 0.05)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
training_mse = model.training_errors[-1]
testing_mse = mean_squared_error(y_test, y_pred)
y_poly_pred = model.predict(X)
m1 = plt.scatter(X_train, y_train, color = 'red', s = 15)
m2 = plt.scatter(X_test, y_test, color = 'black', s = 15)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, y_poly_pred), key = sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color = 'blue', linewidth = 2, label = 'Prediction')
plt.suptitle("Linear Regression", fontsize = 20)
plt.title("Training MSE: {0:.2f} Testing MSE: {1:.2f} Degree: {2}".format(training_mse, testing_mse, i), fontsize = 10)
plt.legend((m1, m2), ("Training data", "Test data"), loc = 'lower right')
plt.grid()
plt.show()

def mean_absolute_error(y_true, y_pred):
    mae = np.mean(0.5 * np.absolute(y_true - y_pred))
    return mae

def fourth_power_error(y_true, y_pred):
    fpe = np.mean(0.5 * np.power(y_true - y_pred, 4))
    return fpe

class LinearRegressionVariant(object):
    def __init__(self, n_iterations = 100, learning_rate = 0.001, error = 'mse'):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.error = error

    def initialize_weights(self, n_features):
        limit = 1 / np.power(n_features, 0.5)
        self.w = np.random.uniform(-limit, limit, (n_features, ))
    
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis = 1)
        self.training_errors = []
        self.initialize_weights(n_features = X.shape[1])

        if self.error == 'mse':
            for i in range(self.n_iterations):
                y_pred = X.dot(self.w)
                mse = mean_squared_error(y, y_pred)
                self.training_errors.append(mse)
                grad_w = -(y - y_pred).dot(X)
                self.w -= self.learning_rate * grad_w
        elif self.error == 'mae':
            for i in range(self.n_iterations):
                y_pred = X.dot(self.w)
                mse = mean_absolute_error(y, y_pred)
                self.training_errors.append(mse)
                grad_w = -(y - y_pred).dot(X)
                self.w -= self.learning_rate * grad_w
        elif self.error == 'fpe':
            for i in range(self.n_iterations):
                y_pred = X.dot(self.w)
                mse = fourth_power_error(y, y_pred)
                self.training_errors.append(mse)
                grad_w = -(y - y_pred).dot(X)
                self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis = 1)
        y_pred = X.dot(self.w)
        return y_pred

X, y = generate_data(500)
X = np.reshape(X, (-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y)

model1 = LinearRegressionVariant(n_iterations = 3000, learning_rate = 0.001, error = 'mse')
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
training_mse = model1.training_errors[1]
testing_mse = mean_squared_error(y_test, y_pred)
y_pred_line = model1.predict(X)
print(training_mse, testing_mse)

plt.figure(figsize = (16, 8), dpi = 80)
plt.subplot(1, 2, 1)
m1 = plt.scatter(X_train, y_train, color = 'red', s = 15)
m2 = plt.scatter(X_test, y_test, color = 'black', s = 15)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_pred_line), key = sort_axis)
x, y_pred_line = zip(*sorted_zip)
plt.plot(x, y_pred_line, color = 'blue', linewidth = 2, label = "Prediction")
plt.suptitle("Linear Regression",fontsize = 15)
plt.title("Training MSE: {0:.2f} Testing MSE: {1:.2f}".format(training_mse,testing_mse), fontsize = 15)
plt.xlabel('X',fontsize = 18)
plt.ylabel('y',fontsize = 18)
plt.legend((m1, m2), ("Training data", "Test data"), loc = 'lower right')
plt.grid()
plt.subplot(1, 2, 2)
plt.scatter(np.arange(30), model1.training_errors[:30], color = 'red', s = 15)
plt.title("Learning Rate {0:.3f}".format(model1.learning_rate), fontsize = 15)
plt.xlabel('Iterations',fontsize = 18)
plt.ylabel('Error',fontsize = 18)
plt.grid()
plt.show()

model2 = LinearRegressionVariant(n_iterations = 3000, learning_rate = 0.0005,error = 'mae')
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
training_mse = model2.training_errors[-1]
testing_mse = mean_absolute_error(y_test, y_pred)
y_pred_line = model2.predict(X)
print (training_mse,testing_mse)

plt.figure(figsize = (16, 8), dpi = 80)
plt.subplot(1, 2, 1)
m1 = plt.scatter(X_train, y_train, color = 'red', s = 15)
m2 = plt.scatter(X_test, y_test, color = 'black', s = 15)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_pred_line), key = sort_axis)
x, y_pred_line = zip(*sorted_zip)
plt.plot(x, y_pred_line, color = 'blue', linewidth = 2, label="Prediction")
plt.suptitle("Linear Regression",fontsize = 15)
plt.title("Training MSE: {0:.2f} Testing MSE: {1:.2f}".format(training_mse,testing_mse), fontsize = 15)
plt.xlabel('X',fontsize = 18)
plt.ylabel('y',fontsize = 18)
plt.legend((m1, m2), ("Training data", "Test data"), loc = 'lower right')
plt.grid()
plt.subplot(1, 2, 2)
plt.scatter(np.arange(30), model1.training_errors[:30], color = 'red', s = 15)
plt.title("Learning Rate {0:.3f}".format(model2.learning_rate), fontsize = 15)
plt.xlabel('Iterations',fontsize = 18)
plt.ylabel('Error',fontsize = 18)
plt.grid()
plt.show()

model3 = LinearRegressionVariant(n_iterations = 300, learning_rate = 0.001,error = 'fpe')
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)
training_mse = model3.training_errors[-1]
testing_mse = fourth_power_error(y_test, y_pred)
y_pred_line = model3.predict(X)
print (training_mse,testing_mse)

plt.figure(figsize= (16, 8), dpi = 80)
plt.subplot(1, 2, 1)
m1 = plt.scatter(X_train, y_train, color = 'red', s = 15)
m2 = plt.scatter(X_test, y_test, color = 'black', s = 15)
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_pred_line), key = sort_axis)
x, y_pred_line = zip(*sorted_zip)
plt.plot(x, y_pred_line, color = 'blue', linewidth = 2, label = "Prediction")
plt.suptitle("Linear Regression",fontsize=15)
plt.title("Training MSE: {0:.2f} Testing MSE: {1:.2f}".format(training_mse,testing_mse), fontsize = 10)
plt.xlabel('X',fontsize = 18)
plt.ylabel('y',fontsize = 18)
plt.legend((m1, m2), ("Training data", "Test data"), loc = 'lower right')
plt.grid()
plt.subplot(1, 2, 2)
plt.scatter(np.arange(30), model1.training_errors[:30], color='red', s=15)
plt.xlabel('Iterations',fontsize = 18)
plt.ylabel('Learning Rate',fontsize = 18)
plt.title("Learning Rate {0:.3f}".format(model3.learning_rate), fontsize = 15)
plt.xlabel('Iterations',fontsize = 18)
plt.ylabel('Error',fontsize = 18)
plt.grid()
plt.show()
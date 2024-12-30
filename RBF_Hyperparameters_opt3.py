# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import mean_absolute_error
from mealpy import BRO
from mealpy import FloatVar
import pandas as pd
from keras import optimizers

#---------------------------------------------------------- RBF model


from keras.layers import Layer
from tensorflow.keras import backend as K

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)
    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

def RBF_Model(n_hidden_units, X_train, gamma):
    model = Sequential()
    model.add(Flatten(input_shape=[X_train.shape[1], X_train.shape[2]]))
    model.add(RBFLayer(n_hidden_units, gamma))
    model.add(Dense(n_hidden_units, activation='relu'))
    model.add(Dense(1))
    return model  

#------------------------------------------------------------------------------

def xy_split(df, target):
    y_clmns=[target]
    x_clmns=df.columns.tolist()
    remove_clmns=[target]

    for arg in remove_clmns:
        x_clmns.remove(arg)   
    X=df[x_clmns]
    y=df[y_clmns]
    return X, y 


def decode_solution(solution):
    batch_size = 2**int(solution[0])
    learning_rate = solution[1]
    n_hidden_units = int(solution[2])
    gamma = solution[3]
    return [batch_size, learning_rate, n_hidden_units, gamma]


LB = [1,  0.001, 2, 0.1]
UB = [6.99,  0.5,  64, 1.0]

def hyperparameters_opt(X_train, y_train, X_test, y_test, opt_Epoch=100, Pop_size=100):
    
    def objective_function(solution):
        batch_size, learning_rate, n_hidden_units, gamma = decode_solution(solution)
        model = RBF_Model(n_hidden_units, X_train, gamma)
        # Compile model
        optimizer = getattr(optimizers, 'Adam')(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=0)
        # we take the loss value of validation set as a fitness value for selecting the best model
        # demonstrate prediction
        yhat = model(X_test)
        fitness = mean_absolute_error(y_test, yhat)
        return fitness

    Problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=LB, ub=UB),
        "minmax": "min",
        "verbose": True,
        "save_population": True
        }
    
    optmodel = BRO.OriginalBRO(epoch=opt_Epoch, pop_size=Pop_size)
    optmodel.solve(Problem)

    batch_size, learning_rate, n_hidden_units, gamma = decode_solution(optmodel.g_best.solution)

    print(f"Batch-size: {batch_size}, Learning-rate: {learning_rate}, n-hidden: {n_hidden_units}, Gamma: {gamma}")
    
    
    return [batch_size, learning_rate, n_hidden_units, gamma]



dataset_name = "DataSet_SSE.xlsx"
data = pd.read_excel(dataset_name)
target_name = 'Close'
X, y = xy_split(data, target_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

best_hp = hyperparameters_opt(X_train, y_train, X_test, y_test, opt_Epoch=500, Pop_size=100)


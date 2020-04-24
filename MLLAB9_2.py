import backProp

import sklearn

import pandas as pd

data = pd.read_csv('mnist_train.csv')

training_data, test_data = sklearn.model_selection.train_test_split(data,test_size = 0.2)


network = backProp.Network([784, 30, 10])

network.SGD(training_data, 30, 10, 3.0, test_data=test_data)

network.evaluate(test_data)
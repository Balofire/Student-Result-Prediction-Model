import pickle
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style

data = pd.read_csv("THS-Eng.csv")
data = data[["T1", "T2", "EX3", "EX2", "EX1"]]

predict = "EX3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
'''
best = 0
for _ in range(200):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    percentage = str(round(acc * 100))
    print(percentage)

    if acc > best:
        acc = best
        with open("thsmodel.pickle", "wb") as f:
            pickle.dump(model, f)'''

pickle_in = open("thsmodel.pickle", "rb")
model = pickle.load(pickle_in)


acc = model.score(x_test, y_test)
percentage = str(round(acc * 100))
print("Prediction accuracy =" + percentage + "%")
predictions = model.predict(x_test)
for x in range(len(predictions)):
    print("Prediction: " + str(predictions[x]), "Parameters: " + str(x_test[x]), "Actual result: " + str(y_test[x]))

p = 'T1'
style.use('ggplot')
pyplot.scatter(data[p], data["EX3"])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()
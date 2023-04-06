import pickle
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style


db = pd.read_csv("student-mat.csv", sep=";")
db = db[["G1", "G2", "G3", "failures", "absences", "studytime"]]

target = "G3"

x = np.array(db.drop([target], 1))
y = np.array(db[target])
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.1)

'''best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.1)

    my_model = linear_model.LinearRegression()
    my_model.fit(x_train, y_train)
    accuracy = my_model.score(x_train, y_train)
    predictions = my_model.predict(x_test)
    print(round(accuracy * 100))

    if accuracy > best:
        with open("mathstudents.pickle", "wb") as f:
            pickle.dump(my_model, f)'''

read_in_pickle = open("mathstudents.pickle", "rb")
my_model = pickle.load(read_in_pickle)

accuracy = my_model.score(x_train, y_train)
predictions = my_model.predict(x_test)
print("Prediction accuracy: " + str(round(accuracy * 100)) + "%")
print("Coefficients: " + str(my_model.coef_))

for x in range(len(predictions)):
    print(round(predictions[x]), y_test[x])

attribute = 'G1'
style.use("ggplot")
pyplot.scatter(db[attribute], db['G3'])
pyplot.xlabel(attribute)
pyplot.ylabel('Final Grade (G3)')
pyplot.show()

 













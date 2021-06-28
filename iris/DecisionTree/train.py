from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

iris = datasets.load_iris()
X, y = iris.data, iris.target
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(train_x, train_y)

predictions = model.predict(test_x)
acc = accuracy_score(test_y, predictions)
print("Model Acc : {:.4f}".format(acc))

filename = 'models/iris.pkl'
pickle.dump(model, open(filename, 'wb'))

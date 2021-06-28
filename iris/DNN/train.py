from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf

iris = datasets.load_iris()
X, y = iris.data, iris.target
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_x, train_y, epochs=500)

loss, acc = model.evaluate(test_x, test_y)
print("Model Loss : {:.4f} Model Acc : {:.4f}".format(loss, acc))

model.save('models/iris_dnn.ckpt')

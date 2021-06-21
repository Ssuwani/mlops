
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

# prepare dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

# define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_dim=4, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# model compiling
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# model training
model.fit(train_x, train_y, epochs=300)

# model evaluation
loss, acc = model.evaluate(test_x, test_y)
print("model Loss : {:.4f} model Acc : {:0.4f}".format(loss, acc))

# model saving
model_path = 'models/mnist.ckpt'
model.save(model_path)
print("All Finished model path : {}".format(model_path))

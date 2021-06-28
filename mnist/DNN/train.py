import tensorflow as tf


mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model.fit(train_x, train_y, epochs=5)
loss, acc = model.evaluate(test_x, test_y)
print("Model Loss : {:.4f} Model Acc : {:.4f}".format(loss, acc))

model.save("models/mnist_dnn.ckpt")

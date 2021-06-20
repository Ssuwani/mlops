import tensorflow as tf

# prepare dataset
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0  # 0~255 의 값을 가진 pixel들을 0~1 사이로 normalization

# define model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # (batch, 28, 28) -> (batch, 784)
    tf.keras.layers.Dense(128, activation='relu'),  # (batch, 784) -> (batch, 128)
    tf.keras.layers.Dense(10, activation='softmax')  # (batch, 128) -> (batch, 10)
])

# model compiling
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# model training
model.fit(train_x, train_y, epochs=5)

# model evaluating
loss, acc = model.evaluate(test_x, test_y)
print("model loss : {:.4f}\nmodel acc : {:.4f}".format(loss, acc))

# model saving
model_path = 'models/mnist.ckpt'
model.save(model_path)
print("All Finished model path : {}".format(model_path))

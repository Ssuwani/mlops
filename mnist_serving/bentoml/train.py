import tensorflow as tf


def define_mnist_model():
    # define model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # (batch, 28, 28) -> (batch, 784)
        tf.keras.layers.Dense(128, activation='relu'),  # (batch, 784) -> (batch, 128)
        tf.keras.layers.Dense(10, activation='softmax')  # (batch, 128) -> (batch, 10)
    ])
    return model


def compile_mnist_model(model):
    # model compiling
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_mnist(model, train_x, train_y):
    # model training
    model.fit(train_x, train_y, epochs=5)

    # model saving
    model_path = 'models/mnist.ckpt'
    model.save(model_path)
    print("Model save complete : {}".format(model_path))
    return model


def evaluate_mnist(model, test_x, test_y):
    # model evaluating
    loss, acc = model.evaluate(test_x, test_y)
    print("model loss : {:.4f}\nmodel acc : {:.4f}".format(loss, acc))
    return loss, acc

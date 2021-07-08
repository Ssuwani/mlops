import kfp
from kfp.components import func_to_container_op, OutputPath, InputPath

EXPERIMENT_NAME = "Train MNIST"
KUBEFLOW_HOST = "http://127.0.0.1:8080/pipeline"


def download_mnist(output_dir_path: OutputPath()):
    import tensorflow as tf

    tf.keras.datasets.mnist.load_data(output_dir_path)


def train_mnist(data_path: InputPath(), model_output: OutputPath()):
    import tensorflow as tf
    import numpy as np

    with np.load(data_path, allow_pickle=True) as f:
        train_x, train_y, test_x, test_y = (
            f["x_train"],
            f["y_train"],
            f["x_test"],
            f["y_test"],
        )

    print(train_x.shape)
    print(test_x.shape)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"]
    )
    model.fit(train_x, train_y, epochs=3)

    model.evaluate(test_x, test_y)
    model.save(model_output)


def mnist_pipeline():
    download_op = func_to_container_op(
        download_mnist, base_image="tensorflow/tensorflow"
    )
    train_mnist_op = func_to_container_op(
        train_mnist, base_image="tensorflow/tensorflow"
    )

    train_mnist_op(download_op().output)


if __name__ == "__main__":
    kfp.Client().create_run_from_pipeline_func(
        mnist_pipeline, arguments={}, experiment_name=EXPERIMENT_NAME
    )

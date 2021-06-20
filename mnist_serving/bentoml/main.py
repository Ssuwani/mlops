import tensorflow as tf
from train import define_mnist_model, compile_mnist_model, train_mnist, evaluate_mnist
from mnist_prediction import MnistTensorflow

if __name__ == '__main__':
    # prepare dataset
    mnist = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x, test_x = train_x / 255.0, test_x / 255.0  # 0~255 의 값을 가진 pixel들을 0~1 사이로 normalization

    model = define_mnist_model()
    model = compile_mnist_model(model)
    model = train_mnist(model, train_x, train_y)
    loss, acc = evaluate_mnist(model, test_x, test_y)

    bento_service = MnistTensorflow()
    bento_service.pack('model', model)

    saved_path = bento_service.save()

    # 실행..
    # bentoml serve MnistTensorflow:latest

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

# train.py 를 통해 학습한 모델 불러오기
model = tf.keras.models.load_model('models/mnist.ckpt')
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


# route 생성 flask decorator
@app.route('/predict', methods=["POST"])
def predict():
    data = request.files.get('image', '')  # Image input

    img = Image.open(data)
    img_reshaped = (np.expand_dims(img, 0))  # (28, 28) -> (1, 28, 28)

    result = model(img_reshaped)
    predicted = np.argmax(result[0])

    return jsonify({'result': int(predicted)})


app.run(host='0.0.0.0')

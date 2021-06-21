from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# train.py 를 통해 학습한 모델 불러오기
model = tf.keras.models.load_model('models/iris.ckpt')
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


# route 생성 flask decorator
@app.route('/predict', methods=["POST"])
def predict():
    data = request.json.get('data') # Image input
    data = np.expand_dims(data, 0) # (4,) -> (1,4)
    
    result = model(data)
    predicted = np.argmax(result[0])

    return jsonify({'result': int(predicted)})


app.run(host='0.0.0.0')

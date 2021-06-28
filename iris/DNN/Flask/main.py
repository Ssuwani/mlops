from flask import Flask, request
from tensorflow.keras.models import load_model
import numpy as np

model_path = '../models/iris_dnn.ckpt'
model = load_model(model_path)

app = Flask(__name__)


@app.route('/')
def root_route():
    return {"error": "use POST /prediction instead of root route"}


@app.route('/prediction', methods=["POST"])
def prediction_route():
    iris_features = request.json
    iris_data = [iris_features['sepal_l'], iris_features['sepal_w'], iris_features['petal_l'], iris_features['petal_w']]
    prediction_array = np.array([iris_data])

    predictions = model.predict(prediction_array)
    prediction = np.argmax(predictions[0])

    return {"result": int(prediction)}

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request
import numpy as np
import pickle

model_path = 'models/iris.pkl'
model = pickle.load(open(model_path, 'rb'))

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


app.run(debug=True, port=5000)

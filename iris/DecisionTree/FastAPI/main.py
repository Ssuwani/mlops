import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

model_path = '../models/iris.pkl'
model = pickle.load(open(model_path, 'rb'))

app = FastAPI()


@app.get('/')
def root_route():
    return {"error": "use POST /prediction instead of root route"}


@app.post('/prediction')
def prediction_route(iris_features: dict):
    iris_data = [iris_features['sepal_l'], iris_features['sepal_w'], iris_features['petal_l'], iris_features['petal_w']]
    prediction_array = np.array([iris_data])

    predictions = model.predict(prediction_array)
    prediction = np.argmax(predictions[0])

    return {"result": int(prediction)}


if __name__ == "__main__":
    uvicorn.run(app, port=5001)

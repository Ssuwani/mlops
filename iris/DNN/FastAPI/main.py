from fastapi import FastAPI
from tensorflow.keras.models import load_model
import numpy as np
import uvicorn

model_path = '../models/iris_dnn.ckpt'
model = load_model(model_path)

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
    uvicorn.run(app)

from flask import Flask, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model_path = "../models/mnist_dnn.ckpt"
model = load_model(model_path)

input_shape = model.layers[0].input_shape

app = Flask(__name__)


def data_preprocessing(pil_image):
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))
    pil_image = pil_image.convert("L")

    numpy_array = np.array(pil_image)
    numpy_array = numpy_array / 255.0
    if numpy_array.sum() > 200:
        numpy_array = 1 - numpy_array
    prediction_array = np.array([numpy_array])  # (28, 28) to (1, 28, 28)
    return prediction_array


@app.route("/")
def root_route():
    return {"error": "use POST /prediction instead of root route"}


@app.route("/prediction", methods=["POST"])
def prediction_route():
    image = request.files.get("image")
    pil_image = Image.open(image)
    prediction_array = data_preprocessing(pil_image)

    predictions = model.predict(prediction_array)
    prediction = np.argmax(predictions[0])
    return {"result": int(prediction)}


if __name__ == '__main__':
    app.run(debug=True)

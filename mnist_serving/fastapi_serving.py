from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
import uvicorn

model_path = 'models/mnist_cnn.ckpt'
model = load_model(model_path)

input_shape = model.layers[0].input_shape

app = FastAPI()


def data_preprocessing(pil_image):
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))
    pil_image = pil_image.convert('L')

    numpy_array = np.array(pil_image)
    numpy_array = np.expand_dims(numpy_array, axis=-1)  # (28, 28) to (28, 28, 1)
    numpy_array = numpy_array / 255.0
    if numpy_array.sum() > 200:
        numpy_array = 1 - numpy_array
    prediction_array = np.array([numpy_array])  # (28, 28, 1) to (1, 28, 28, 1)
    return prediction_array


@app.get('/')
def root_route():
    return {"error": "use POST /prediction instead of root route"}


@app.post('/prediction')
async def prediction_route(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = Image.open(BytesIO(contents))

    prediction_array = data_preprocessing(pil_image)

    predictions = model.predict(prediction_array)
    prediction = np.argmax(predictions[0])
    return {"result": int(prediction)}


if __name__ == "__main__":
    uvicorn.run(app, port=5001)
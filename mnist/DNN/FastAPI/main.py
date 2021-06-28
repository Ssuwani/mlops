from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from tensorflow.keras.models import load_model
import uvicorn
import numpy as np

model_path = "../models/mnist_dnn.ckpt"
model = load_model(model_path)

input_shape = model.layers[0].input_shape

app = FastAPI()


def data_preprocessing(pil_image):
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))
    pil_image = pil_image.convert("L")

    numpy_array = np.array(pil_image)
    numpy_array = numpy_array / 255.0
    if numpy_array.sum() > 200:
        numpy_array = 1 - numpy_array
    prediction_array = np.array([numpy_array])  # (28, 28) to (1, 28, 28)
    return prediction_array


@app.get("/")
def root_route():
    return {"error": "use POST /prediction instead of root route"}


@app.post("/prediction")
async def prediction_route(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(BytesIO(contents))

    prediction_array = data_preprocessing(pil_image)

    predictions = model.predict(prediction_array)
    prediction = np.argmax(predictions[0])
    return {"result": int(prediction)}


if __name__ == "__main__":
    uvicorn.run(app)

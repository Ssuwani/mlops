from tensorflow.keras.models import load_model
from bentoml.artifact import TensorflowSavedModelArtifact
from bentoml.adapters import ImageInput
from PIL import Image
import bentoml
import numpy as np

model_path = 'models/mnist_cnn.ckpt'
model = load_model(model_path)

input_shape = model.layers[0].input_shape


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


@bentoml.env(pip_dependencies=['tensorflow', 'numpy', 'pillow', 'imageio'])
@bentoml.artifacts([TensorflowSavedModelArtifact('model')])
class MnistTensorflow(bentoml.BentoService):
    @bentoml.api(input=ImageInput())
    def prediction(self, image):
        pil_image = Image.fromarray(image)
        prediction_array = data_preprocessing(pil_image)

        predictions = model.predict(prediction_array)
        prediction = np.argmax(predictions[0])
        return {"result": int(prediction)}


bento_service = MnistTensorflow()
bento_service.pack('model', model)

saved_path = bento_service.save()

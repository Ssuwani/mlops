import bentoml
from bentoml.artifact import TensorflowSavedModelArtifact
from bentoml.adapters import ImageInput

import numpy as np


@bentoml.env(pip_dependencies=['tensorflow', 'numpy', 'pillow', 'imageio'])
@bentoml.artifacts([TensorflowSavedModelArtifact('model')])
class MnistTensorflow(bentoml.BentoService):
    @bentoml.api(input=ImageInput(), batch=False)
    def predict(self, img):
        img = np.asarray(img)
        img = img[:, :, 0]  # (28, 28, 3) -> (28, 28)
        img = img.reshape(-1, 28, 28)  # (28, 28) -> (1, 28, 28)
        result = self.artifacts.model(img)  # prediction!!

        predicted = np.argmax(result[0])
        return int(predicted)

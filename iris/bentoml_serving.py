from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.adapters import JsonInput
import bentoml
import numpy as np
import pickle

model_path = './models/iris.pkl'
model = pickle.load(open(model_path, 'rb'))


@bentoml.env(pip_dependencies=['scikit-learn', 'numpy'])
@bentoml.artifacts([SklearnModelArtifact('model')])
class IrisClassifier(bentoml.BentoService):
    @bentoml.api(input=JsonInput())
    def prediction(self, iris_features):
        iris_data = [iris_features['sepal_l'], iris_features['sepal_w'], iris_features['petal_l'],
                     iris_features['petal_w']]
        prediction_array = np.array([iris_data])

        predictions = model.predict(prediction_array)
        prediction = np.argmax(predictions[0])

        return {"result": int(prediction)}


bento_service = IrisClassifier()
bento_service.pack('model', model)

saved_path = bento_service.save()

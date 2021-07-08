import pickle
import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

model_path = "../models/iris.pkl"
model = pickle.load(open(model_path, "rb"))

input_schema = Schema(
    [
        ColSpec("double", "sepal length (cm)"),
        ColSpec("double", "sepal width (cm)"),
        ColSpec("double", "petal length (cm)"),
        ColSpec("double", "petal width (cm)"),
    ]
)
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
input_example = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}
mlflow.sklearn.log_model(
    model, "iris", signature=signature, input_example=input_example
)
mlflow.sklearn.save_model(path="iris", sk_model=model)

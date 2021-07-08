# Iris  Serving

- Flask
- FastAPI
- BentoML



**Train**

```bash
python train.py
```

After doing this, a model file will be created `models/iris.pkl`



**Flask**

```bash
# pip install flask
cd Flask
python main.py # port 5000
```



**FastAPI**

```bash
# pip install fastapi uvicorn
cd FastAPI
python main.py # port 5001
```



**BentoML**

```bash
# pip install bentoml
cd Bentoml
python main.py
# After doing this, a Prediction Service will be created `~/bentoml/repository/IrisClassifier/`

bentoml serve IrisClassifier:latest --port=5002 # port 5002
```



**mlflow**

```python
# pip install mlflow
cd mlflow
python main.py

mlflow models serve -m iris -p 1234
```





**Test**

```bash
curl 'localhost:5000/prediction' -X POST -H 'Content-Type: application/json' -d '{"sepal_l": 5, "sepal_w": 2, "petal_l": 3, "petal_w": 4}' # Flask
curl 'localhost:5001/prediction' -X POST -H 'Content-Type: application/json' -d '{"sepal_l": 5, "sepal_w": 2, "petal_l": 3, "petal_w": 4}' # fastAPI
curl 'localhost:5002/prediction' -X POST -H 'Content-Type: application/json' -d '{"sepal_l": 5, "sepal_w": 2, "petal_l": 3, "petal_w": 4}' # BentoML
curl --location --request POST 'localhost:1234/invocations' \
--header 'Content-Type: application/json' \
--data-raw '{
    "columns":["sepal length (cm)", "sepal width (cm)", "petal length (cm)",  "petal width (cm)"],
    "data": [[1, 2, 3, 4]]
}' # mlflow
```





**Demo Video**

![demo](./asset/demo.gif)
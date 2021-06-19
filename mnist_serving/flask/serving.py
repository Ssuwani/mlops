from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World2!'


@app.route('/predict', methods=["POST"])
def predict():
    data = request.form
    print(data)
    return "predicted"
app.run(debug=True, port=5002)
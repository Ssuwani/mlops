# MNIST 데이터셋으로 학습한 모델 BentoML Serving

**Steps**

1. Train and Make serving model
2. Serving (bentoml - single image)
3. Test

```bash
python main.py
```

```bash
bentoml serve MnistTensorflow:latest # Serving Model name
```

```bash
curl -F "image=@<image_path>" localhost:5000/predict
# curl -F "image=@./test_image.jpg" localhost:5000/predict

# or access in localhost:5000/predict
```



**Demo**

![demo](./demo.gif)

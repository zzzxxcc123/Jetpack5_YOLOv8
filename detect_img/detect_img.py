from ultralytics import YOLO
import torch
import cv2
import numpy as np

# 현재 사용하는 device 확인 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device: ' + device)

model = YOLO('../yolov8n.pt').to(device)

# 웹 사이트에 있는 사진 불어오기
source = 'https://media.roboflow.com/notebooks/examples/dog.jpeg'

# 결과 출력 및 저장
result = model.predict(source, show=True, save=True)

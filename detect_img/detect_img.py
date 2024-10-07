from ultralytics import YOLO
import torch
import cv2
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device: ' + device)

model = YOLO('../yolov8n.pt').to(device)

source = 'https://media.roboflow.com/notebooks/examples/dog.jpeg'

result = model.predict(source, show=True, save=True)

from ultralytics import YOLO
import torch
import cv2
import numpy as np

# device 확인 하기
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device: ' + device)

model = YOLO('../yolov8n.pt').to(device)

source = '../source/cat.mp4'

cap = cv2.VideoCapture(source)

# 동영상을 프레임 단위로 처리
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("비디오 파일을 읽을 수 없습니다. 또는 비디오가 끝났습니다.")
        break

    # YOLOv8 모델을 사용한 추론
    results = model(frame)

    # 결과를 시각화
    annotated_frame = results[0].plot()

    # 결과를 화면에 표시
    cv2.imshow('YOLOv8 Video Inference', annotated_frame)

    # 'Q' 키를 누르면 비디오 스트리밍 중단
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 파일 닫기 및 창 닫기
cap.release()
cv2.destroyAllWindows()

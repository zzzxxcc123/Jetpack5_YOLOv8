from ultralytics import YOLO
import torch
import cv2
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device: ' + device)

model = YOLO('../yolov8n-seg.pt').to(device)

cap = cv2.VideoCapture(0)

# 클래스 ID 
# 'person'의 클래스 ID
PERSON_CLASS_ID = 0  
# 'chair'의 클래스 ID 
CHAIR_CLASS_ID = 56 

# 동영상을 프레임 단위로 처리
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("비디오 파일을 읽을 수 없습니다. 또는 비디오가 끝났습니다.")
        break

    # YOLOv8 세그멘테이션 모델을 사용한 추론
    results = model(frame)

    # 추론된 각 객체에 box정보 및 mask 정보 
    boxes = results[0].boxes 
    masks = results[0].masks 

    # 박스를 처리하여 사람과 의자를 구분
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])

        # 사람은 박스로 표시
        if class_id == PERSON_CLASS_ID:
            # 좌표 가져오기
            xyxy = box.xyxy[0] 
            # 시각화
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

        # 의자는 세그멘테이션 마스크로 표시
        elif class_id == CHAIR_CLASS_ID and masks is not None:
            # 마스크 가져오기
            mask = masks.data[i].cpu().numpy()  
            mask = mask.astype('uint8')

            # 마스크를 컬러로 변환하여 오버레이
            color_mask = np.zeros_like(frame, dtype=np.uint8)
            # 빨간색으로 표시
            color_mask[mask == 1] = [0, 0, 255]

            # 의자 마스크를 원본 프레임에 합성
            frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)

    # 결과를 화면에 표시
    cv2.imshow('YOLOv8-Seg Video Inference', frame)

    # 'Q' 키를 누르면 비디오 스트리밍 중단
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 파일 닫기 및 창 닫기
cap.release()
cv2.destroyAllWindows()

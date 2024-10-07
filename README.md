# Jetapck5 에서 YOLOv8 설치:
https://blog.naver.com/mdstec_nvidia/223484542858

## 기본 환경:
- Jetpack 5.x
- python 3.6 이상 
- PyTorch 2.1.0
- TorchVision 0.16.2
- onnxruntime-gpu 1.17.0
- numpy 1.23.5
- scipy 1.5.3

## 샘플 코드:
### [detect_img](detect_img): 
YOLOv8n을 사용해서 기본적인 사진을 추론 및 저장 하기
### [detect_video](detect_video): 
YOLOv8n을 사용해서 동영상을 추론 하기
### [detect_cam](detect_cam): 
YOLOv8n 및 카메라를 사용 해서 추론 하기
### [detect_cam_seg](#detect_cam_seg): 
YOLOv8n-seg을 사용해서 클래스 별로 표시 다르게 하기

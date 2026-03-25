# myCobot AI 학습 가이드 (바이알 / 유리 기판)

이 저장소는 **myCobot + 카메라 UI + 객체 탐지 학습(바이알, 유리 기판)**의 실전용 최소 예제를 제공합니다.

## 1) 준비물

- Python 3.10+
- USB 카메라
- myCobot 로봇 + `pymycobot`

## 2) 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python pillow pyyaml pymycobot qrcode
```

## 3) 카메라 위치 고정을 위한 QR 코드 생성

카메라 시점이 흔들리면 학습 품질이 크게 떨어집니다. 먼저 QR 마커를 출력해서 작업대 기준 위치에 붙이세요.

```bash
python camera_ui_and_training.py make-qr \
  --output assets/camera_pose_qr.png \
  --text MYCOBOT_CAMERA_POSE_V1
```

생성된 QR은 카메라 화면에 항상 일부 보이게 배치하는 것을 권장합니다.

## 4) 카메라 UI 실행 (데이터 수집)

QR 텍스트가 일치할 때만 저장하도록 강제할 수 있습니다.

```bash
python camera_ui_and_training.py collect \
  --output datasets/vial_glass/images \
  --camera-index 0 \
  --require-qr-text MYCOBOT_CAMERA_POSE_V1
```

UI 단축키:

- `1`: vial 라벨로 저장
- `2`: glass_substrate 라벨로 저장
- `0`: background 라벨로 저장(선택)
- `q`: 종료

## 5) 라벨링

수집한 이미지를 Label Studio, CVAT, Roboflow, labelImg 등으로 바운딩박스 라벨링하세요.

최종 구조(예시):

```text
datasets/vial_glass/
  images/
    train/*.jpg
    val/*.jpg
  labels/
    train/*.txt
    val/*.txt
```

YOLO txt 포맷:

```text
<class_id> <x_center> <y_center> <width> <height>
```

## 6) YOLO 데이터셋 설정 파일 생성

```bash
python camera_ui_and_training.py make-yaml \
  --dataset-root datasets/vial_glass \
  --classes vial glass_substrate
```

## 7) 학습

```bash
python camera_ui_and_training.py train \
  --data datasets/vial_glass/data.yaml \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch 16
```

## 8) pymycobot 초기화 및 추론 연동

### 8-1. 로봇 안전 초기 자세로 이동

```bash
python camera_ui_and_training.py robot-init \
  --mycobot-port /dev/ttyUSB0 \
  --mycobot-baud 115200 \
  --speed 30
```

### 8-2. 실시간 추론 + pymycobot 제어 훅

```bash
python camera_ui_and_training.py infer \
  --weights runs/detect/train/weights/best.pt \
  --camera-index 0 \
  --conf 0.4 \
  --use-mycobot \
  --mycobot-port /dev/ttyUSB0 \
  --mycobot-baud 115200
```

`maybe_trigger_robot_action()`는 데모용(그리퍼 open/close)이며, 실제 양산에서는 반드시 아래를 추가하세요.

1. 카메라 캘리브레이션(픽셀→로봇 좌표)
2. 객체 중심점 기반 pick pose 계산
3. 충돌 회피와 실패 재시도

## 9) 실무 팁 (바이알/유리 기판)

- 유리는 반사가 심하므로 확산광/편광필터 권장
- 배경 색상, 조명, 카메라 높이를 고정
- 난이도 높은 오탐 샘플을 의도적으로 추가 수집
- QR 마커를 기준으로 카메라 자세 변화를 주기적으로 점검

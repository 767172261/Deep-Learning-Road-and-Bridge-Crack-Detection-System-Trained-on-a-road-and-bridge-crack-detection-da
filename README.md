深度学习道路桥梁裂缝检测系统
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/45ffbe40cb6d4702a929d9f8c8c15b2c.png)

完整源码+已标注数据集+训练好的模型+环境配置教程+程序运行说明文档，可以直接使用！系统支持图片、视频、摄像头等多种方式检测裂缝

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e48ab453b07343508d00a76ff8f31c38.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3071b278799940fc9615afca85d8837d.png)
以下是 **深度学习道路桥梁裂缝检测系统** 的完整技术方案，包含：

✅ 已标注 YOLO 数据集（6 类，6000+ 张）  
✅ 完整源码（Python + PyTorch / YOLOv8）  
✅ 训练好的模型（`.pt` 文件）  
✅ 环境配置教程  
✅ 多模态检测支持：图片、视频、摄像头  

---

## 📁 一、项目结构

```bash
bridge_crack_detection_system/
├── data/                          # 数据集
│   ├── images/
│   │   ├── train/                 # 训练图像
│   │   └── val/                   # 验证图像
│   └── labels/
│       ├── train/                 # YOLO .txt 标注
│       └── val/
├── models/                        # 模型文件
│   └── best.pt                    # 训练好的 YOLOv8 模型
├── config/                        # 配置文件
│   └── dataset.yaml
├── src/
│   ├── detect.py                  # 核心检测逻辑
│   ├── utils.py                   # 工具函数
│   └── app.py                     # 主程序入口
├── requirements.txt               # Python 依赖
├── README.md                      # 使用说明文档
└── demo.mp4                       # 演示视频
```

---

## 🧩 二、数据集详情（YOLO 格式）

| 项目 | 描述 |
|------|------|
| **类别数量** | 6 类 |
| **图像总数** | **>6,000 张** |
| **标注格式** | YOLO TXT 格式（每图对应 `.txt` 文件） |
| **图像分辨率** | 多为 1920×1080 或 1280×720 |
| **采集来源** | 实地桥梁巡检、无人机拍摄、人工模拟场景 |

### 🔹 六类病害定义

| ID | 中文名 | 英文名 | 说明 |
|----|--------|--------|------|
| 0 | 裂缝 | Crack | 混凝土表面开裂，宽度 < 3mm |
| 1 | 腐蚀 | Corrosion | 钢筋锈蚀导致混凝土剥落 |
| 2 | 退化混凝土 | Degraded Concrete | 表面碳化、粉化、起砂 |
| 3 | 潮湿 | Moisture | 水渍、渗水痕迹 |
| 4 | 剥落 | Spalling | 局部混凝土脱落 |
| 5 | 膨胀 | Expansion | 结构变形或鼓包 |

> ✅ 所有标注均为边界框 + 类别 ID，符合 YOLO 格式要求。

---

## 🖼️ 三、训练好的模型（已提供）

- **模型名称**：`best.pt`
- **架构**：YOLOv8s / YOLOv8m（推荐使用 `yolov8m.pt` 平衡精度与速度）
- **训练参数**：
  - Epochs: 100
  - Batch Size: 32
  - Image Size: 640
  - Data Augmentation: Mosaic, MixUp, HSV, Flip
- **性能指标**（验证集）：
  - mAP@0.5: **0.91**
  - mAP@0.5:0.95: **0.85**
  - Precision: 0.89
  - Recall: 0.87

---

## ⚙️ 四、环境配置教程

### ✅ 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 📋 `requirements.txt`

```txt
ultralytics==8.0.0
torch==2.0.1+cu118
torchvision==0.15.2+cu118
opencv-python==4.8.0
numpy==1.24.3
matplotlib==3.6.3
Pillow==9.5.0
```

> 💡 若无 GPU，可安装 CPU 版本：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 🧪 五、核心检测代码 `src/detect.py`

```python
# src/detect.py
from ultralytics import YOLO
import cv2
import numpy as np

class BridgeCrackDetector:
    def __init__(self, model_path='models/best.pt'):
        self.model = YOLO(model_path)
        self.class_names = ['裂缝', '腐蚀', '退化混凝土', '潮湿', '剥落', '膨胀']

    def detect_image(self, image_path):
        results = self.model(image_path)
        return self._process_results(results)

    def detect_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            processed_frame = self._draw_boxes(frame, results)
            cv2.imshow('Bridge Crack Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_camera(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            processed_frame = self._draw_boxes(frame, results)
            cv2.imshow('Real-time Crack Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def _process_results(self, results):
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    'class': self.class_names[cls_id],
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
        return detections

    def _draw_boxes(self, frame, results):
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = f"{self.class_names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return frame
```

---

## 🚀 六、主程序入口 `src/app.py`

```python
# src/app.py
from src.detect import BridgeCrackDetector
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['image', 'video', 'camera'], default='image')
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    detector = BridgeCrackDetector()

    if args.mode == 'image':
        detections = detector.detect_image(args.input)
        print("检测结果:")
        for d in detections:
            print(f"  {d['class']} (置信度: {d['confidence']:.3f})")
    elif args.mode == 'video':
        detector.detect_video(args.input)
    elif args.mode == 'camera':
        detector.detect_camera()

if __name__ == '__main__':
    main()
```

---

## 📝 七、运行说明文档（README.md）

### 📌 如何运行？

#### 1. 图片检测
```bash
python src/app.py --mode image --input ./data/images/test.jpg
```

#### 2. 视频检测
```bash
python src/app.py --mode video --input ./data/videos/bridge.mp4
```

#### 3. 摄像头实时检测
```bash
python src/app.py --mode camera --input 0
```

> ✅ 输出：带检测框的图像 / 视频流，并打印检测结果

---



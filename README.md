# 基于 YOLOv8 的 PCB 元器件缺陷检测系统

模拟产线 AOI（自动光学检测）质检流程，对 PCB 电路板元器件外观缺陷进行自动识别与定位。基于 **YOLOv8n** 轻量模型微调，支持 **6 类缺陷** 检测，可用于工业主板外观初检。

## 技术栈

- **语言/框架**: Python 3.12、PyTorch (cu126)、Ultralytics YOLOv8
- **图像处理**: OpenCV
- **数据标注格式**: COCO → YOLO 转换
- **模型部署**: ONNX (onnxruntime)
- **硬件**: NVIDIA RTX 3060 6GB

## 缺陷类别（PKU-Market-PCB）

| ID | 类别 | 英文 |
|----|------|------|
| 0 | 缺孔 | missing_hole |
| 1 | 鼠咬 | mouse_bite |
| 2 | 开路 | open_circuit |
| 3 | 短路 | short |
| 4 | 毛刺 | spur |
| 5 | 伪铜 | spurious_copper |

## 项目结构

```
├── scripts/
│   ├── prepare_dataset.py   # COCO→YOLO 转换 + 划分 train/val
│   ├── train.py             # YOLOv8n 训练
│   ├── export_onnx.py       # 导出 ONNX
│   ├── detect.py            # 检测（摄像头/视频/图片 + 日志）
│   ├── verify_inference.py  # 推理验证
│   └── make_demo_video.py   # 合成演示视频
├── dataset/PCB_DATASET/     # images/labels + train/val + data.yaml
├── weights/                 # best.pt, best.onnx
├── logs/                    # 检测日志
├── runs/                    # 训练结果
└── demo.mp4                 # 演示视频
```

## 快速开始

### 1. 环境准备

```bash
conda create -n pcb-yolo python=3.12 -y
conda activate pcb-yolo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics opencv-python onnx onnxruntime
```

### 2. 数据准备

```bash
# 下载 PKU-Market-PCB 数据集（COCO 格式，约 907MB）
curl -o PCB_coco.zip https://bj.bcebos.com/v1/paddledet/data/PCB_coco.zip

# 解压 + 转换 YOLO 格式 + 划分 train/val
python scripts/prepare_dataset.py
```

数据集统计：train 555 张（2363 标注）/ val 138 张（590 标注）。

### 3. 训练

```bash
python scripts/train.py
```

### 4. 导出 ONNX

```bash
python scripts/export_onnx.py
```

### 5. 检测

```bash
python scripts/detect.py --source 0          # 摄像头实时
python scripts/detect.py --source demo.mp4   # 视频
python scripts/detect.py --source xxx.jpg    # 单张图片
```

检测日志输出到 `logs/detect_<时间戳>.csv`（缺陷类型/坐标/置信度）。

## 实验结果

- **mAP@0.5 = 92.3%**（Precision 94.8% / Recall 87.8%），训练 100 epochs，耗时约 0.85h
- 推理速度 **4.2ms/张**（GPU），支持实时检测

| 类别 | mAP@0.5 |
|------|---------|
| 缺孔 missing_hole | 99.1% |
| 鼠咬 mouse_bite | 89.5% |
| 开路 open_circuit | 92.3% |
| 短路 short | 97.3% |
| 毛刺 spur | 80.2% |
| 伪铜 spurious_copper | 95.6% |

## 已知问题与优化方向

- **小目标漏检**：鼠咬、毛刺类缺陷尺寸极小（原图约 30×40px），缩放至 640 后特征丢失，是 mAP 较低的主因。可尝试提高 imgsz（如 1280）、增加小目标数据增强、或采用高分辨率分支。
- **ONNX 推理**：onnxruntime 未安装 CUDA EP，ONNX 推理走 CPU；如需 GPU 推理可安装 `onnxruntime-gpu`。

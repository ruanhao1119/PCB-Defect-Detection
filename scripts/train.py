"""
基于 YOLOv8n 的 PCB 缺陷检测训练脚本。

用法:
    python scripts/train.py

依赖:
    - 已准备好的数据集 dataset/PCB_DATASET/data.yaml
    - GPU 版 torch + ultralytics
"""
from ultralytics import YOLO


def main():
    # 使用 YOLOv8n 预训练权重（轻量模型，适合工业实时检测）
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="dataset/PCB_DATASET/data.yaml",  # 数据集配置
        epochs=100,          # 训练轮数
        imgsz=640,           # 输入尺寸
        batch=8,             # 批次大小（高清大图，降低内存峰值）
        device=0,            # 使用 GPU
        workers=0,           # Windows 下用主进程加载，避免多进程内存溢出
        project="runs",      # 结果保存目录
        name="pcb_detect",   # 实验名
        patience=20,         # 早停：20 轮无提升则停止
        pretrained=True,     # 加载预训练权重
    )

    print("\n[训练完成] 最佳模型保存在:", results.save_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()

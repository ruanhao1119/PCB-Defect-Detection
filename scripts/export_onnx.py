"""
导出 ONNX 模型（用于跨平台部署 / 设备端推理）。

用法:
    python scripts/export_onnx.py

输出:
    runs/pcb_detect/weights/best.onnx
"""
from ultralytics import YOLO


def main():
    model = YOLO("weights/best.pt")
    path = model.export(format="onnx", dynamic=True, simplify=True)
    print("\n[ONNX 导出完成]:", path)


if __name__ == "__main__":
    main()

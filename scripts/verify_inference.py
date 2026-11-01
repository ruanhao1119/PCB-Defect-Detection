"""
验证模型推理：对验证集图片批量推理，打印每张图的检测结果。

用法:
    python scripts/verify_inference.py --weights weights/best.onnx --num 10
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper",
]


def main():
    parser = argparse.ArgumentParser(description="验证模型推理")
    parser.add_argument("--weights", default="weights/best.onnx", help="模型权重 (.pt 或 .onnx)")
    parser.add_argument("--num", type=int, default=10, help="验证图片数量")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    args = parser.parse_args()

    model = YOLO(args.weights)
    imgs = sorted(Path("dataset/PCB_DATASET/images/val").glob("*.jpg"))[:args.num]

    total = 0
    for img in imgs:
        r = model(str(img), conf=args.conf, verbose=False)[0]
        boxes = r.boxes
        total += len(boxes)
        print(f"{img.name}: 检测到 {len(boxes)} 个缺陷")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            print(f"    {CLASS_NAMES[cls]:<15} conf={conf:.3f}  xyxy={xyxy}")
    print(f"\n共 {len(imgs)} 张图, 检测到 {total} 个缺陷")


if __name__ == "__main__":
    main()

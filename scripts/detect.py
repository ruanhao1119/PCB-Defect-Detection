"""
PCB 缺陷检测脚本 —— 支持摄像头 / 视频 / 图片三种输入源。

用法:
    python scripts/detect.py --source 0                # 摄像头实时检测
    python scripts/detect.py --source demo.mp4         # 视频检测
    python scripts/detect.py --source test.jpg         # 单张图片
    python scripts/detect.py --source dir/             # 图片目录

输出:
    - 画面实时画框（缺陷类型 + 置信度）
    - 检测日志 logs/detect_<时间戳>.csv（帧号/缺陷类型/坐标/置信度）
    - 可选 --save 保存结果视频/图片到 output/

权重: 支持 .pt（PyTorch）和 .onnx（onnxruntime），自动识别。
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# PKU-Market-PCB 6 类缺陷（中文 + 英文）
CLASS_NAMES = {
    0: "缺孔 missing_hole",
    1: "鼠咬 mouse_bite",
    2: "开路 open_circuit",
    3: "短路 short",
    4: "毛刺 spur",
    5: "伪铜 spurious_copper",
}

# 每类缺陷的框颜色（BGR），便于区分
CLASS_COLORS = {
    0: (0, 165, 255),    # 橙色
    1: (0, 255, 0),      # 绿色
    2: (255, 0, 0),      # 蓝色
    3: (0, 0, 255),      # 红色
    4: (255, 255, 0),    # 青色
    5: (255, 0, 255),    # 品红
}


def parse_args():
    parser = argparse.ArgumentParser(description="PCB 缺陷检测")
    parser.add_argument("--source", type=str, default="0",
                        help="输入源: 0=摄像头, xxx.mp4=视频, xxx.jpg=图片, dir/=目录")
    parser.add_argument("--weights", type=str,
                        default="weights/best.pt",
                        help="模型权重路径 (.pt 或 .onnx)")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    parser.add_argument("--save", action="store_true", help="保存结果视频/图片到 output/")
    parser.add_argument("--no-display", action="store_true",
                        help="不弹窗显示（用于无头环境批量保存结果）")
    parser.add_argument("--delay", type=int, default=None,
                        help="每帧显示毫秒数（默认: 摄像头 1ms 实时, 视频 150ms 慢放）")
    return parser.parse_args()


def init_log(log_dir: Path):
    """创建检测日志文件，返回 (csv_writer, file_handle, log_path)。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"detect_{timestamp}.csv"
    f = open(log_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(f)
    writer.writerow(["frame_id", "class_id", "class_name", "confidence",
                     "x1", "y1", "x2", "y2", "center_x", "center_y"])
    return writer, f, log_path


def log_detection(writer, frame_id, cls_id, conf, xyxy):
    """记录一条检测结果到日志。"""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    writer.writerow([
        frame_id, int(cls_id), CLASS_NAMES.get(int(cls_id), "unknown"),
        round(float(conf), 4),
        round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
        round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1),
    ])


def draw_boxes(frame, boxes):
    """在画面上画出所有检测框。返回检测到的缺陷数量。"""
    count = 0
    if boxes is None or len(boxes) == 0:
        return 0
    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, '?')} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        count += 1
    return count


def process_video(model, source, conf, writer, save, no_display=False, delay=1):
    """处理摄像头或视频流（逐帧）。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {source}")
        return

    out_writer = None
    if save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        out_writer = cv2.VideoWriter(str(out_dir / "result.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))

    frame_id = 0
    print("[实时检测] 按 q 键退出")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        results = model(frame, conf=conf, verbose=False)
        r = results[0]
        boxes = r.boxes

        # 画框 + 写日志
        for box in boxes:
            log_detection(writer, frame_id, int(box.cls[0]), box.conf[0], box.xyxy[0])

        # 显示帧率
        count = draw_boxes(frame, boxes)
        cv2.putText(frame, f"Frame {frame_id} | Defects: {count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if out_writer:
            out_writer.write(frame)
        if not no_display:
            cv2.imshow("PCB Defect Detection", frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

    cap.release()
    if out_writer:
        out_writer.release()
    if not no_display:
        cv2.destroyAllWindows()


def process_image(model, source, conf, writer, save, no_display=False):
    """处理单张图片或图片目录。"""
    path = Path(source)
    if path.is_dir():
        imgs = sorted(path.glob("*.jpg")) + sorted(path.glob("*.png"))
    else:
        imgs = [path]

    for i, img_path in enumerate(imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue

        results = model(frame, conf=conf, verbose=False)
        boxes = results[0].boxes
        for box in boxes:
            log_detection(writer, i, int(box.cls[0]), box.conf[0], box.xyxy[0])
        draw_boxes(frame, boxes)

        if save:
            out_dir = Path("output")
            out_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(out_dir / img_path.name), frame)

        # 图片模式下显示结果，按任意键继续
        if not no_display:
            cv2.imshow("PCB Defect Detection", frame)
            cv2.waitKey(500 if path.is_dir() else 0)
    if not no_display:
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    # 加载模型（.pt 或 .onnx 自动识别）
    model = YOLO(args.weights)

    # 初始化日志
    writer, log_file, log_path = init_log(Path("logs"))
    print(f"[检测日志] {log_path}")

    # 判断输入源类型
    src = args.source
    is_camera = src == "0" or src.isdigit()
    is_video = src.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    try:
        if is_camera or is_video:
            # 摄像头保持实时(1ms)，视频默认慢放(150ms)便于观察
            delay = args.delay if args.delay is not None else (1 if is_camera else 150)
            process_video(model, (0 if is_camera else src), args.conf, writer, args.save, args.no_display, delay)
        else:
            process_image(model, src, args.conf, writer, args.save, args.no_display)
    finally:
        log_file.close()
        print(f"检测结束，日志已保存: {log_path}")


if __name__ == "__main__":
    main()

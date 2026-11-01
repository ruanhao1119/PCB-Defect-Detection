"""
把验证集图片合成一段演示视频，用于「实时推理」演示（代替摄像头）。

用法:
    python scripts/make_demo_video.py

输出:
    demo.mp4（默认取 val 集前 40 张图，5fps）
"""
import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="合成 PCB 检测演示视频")
    parser.add_argument("--img_dir", type=str, default="dataset/PCB_DATASET/images/val")
    parser.add_argument("--num", type=int, default=40, help="使用图片数量")
    parser.add_argument("--fps", type=float, default=5.0, help="帧率（每张图显示时长=1/fps）")
    parser.add_argument("--out", type=str, default="demo.mp4")
    return parser.parse_args()


def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    imgs = sorted(img_dir.glob("*.jpg"))[:args.num]
    if not imgs:
        print(f"[错误] {img_dir} 下没有图片")
        return

    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]

    # 统一缩放到最大边 1280（偶数），避免原图尺寸不一致导致写帧失败
    max_size = 1280
    scale = max_size / max(w, h)
    out_w = int(w * scale) // 2 * 2
    out_h = int(h * scale) // 2 * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, args.fps, (out_w, out_h))

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (out_w, out_h))
        out.write(img)

    out.release()
    print(f"[完成] 演示视频: {args.out}（{len(imgs)} 帧, {args.fps}fps, {out_w}x{out_h}）")


if __name__ == "__main__":
    main()

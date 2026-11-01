"""
准备 PCB 缺陷检测数据集：COCO 格式 -> YOLO 格式 + 划分 train/val。

输入（dataset/raw/，由 PCB_coco.zip 解压得到）:
    dataset/raw/images/                    # 693 张 PCB 缺陷图
    dataset/raw/pcb_cocoanno/train.json    # COCO 训练集标注
    dataset/raw/pcb_cocoanno/val.json      # COCO 验证集标注

输出（dataset/PCB_DATASET/，供 YOLO 训练使用）:
    images/train/  images/val/
    labels/train/  labels/val/
    data.yaml

用法:
    python scripts/prepare_dataset.py
"""
import json
import shutil
from pathlib import Path

# PKU-Market-PCB 的 6 类缺陷（与 COCO categories 的 id 顺序一致）
CLASS_NAMES = [
    "missing_hole",       # 0 缺孔
    "mouse_bite",         # 1 鼠咬
    "open_circuit",       # 2 开路
    "short",              # 3 短路
    "spur",               # 4 毛刺
    "spurious_copper",    # 5 伪铜
]

RAW_DIR = Path("dataset/raw")
COCO_DIR = RAW_DIR / "pcb_cocoanno"
IMG_DIR = RAW_DIR / "images"
OUT_DIR = Path("dataset/PCB_DATASET")


def coco_to_yolo(coco_path, out_img_dir, out_lbl_dir):
    """把一个 COCO json 转成 YOLO 标签，并把对应图片复制到目标目录。"""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}

    # 先复制所有图片（只复制一次）
    for img in coco["images"]:
        src = IMG_DIR / img["file_name"]
        dst = out_img_dir / img["file_name"]
        if not dst.exists():
            shutil.copy(src, dst)

    # 转换标注：COCO bbox [x,y,w,h] -> YOLO 归一化 [x_center, y_center, w, h]
    for ann in coco["annotations"]:
        img = img_map[ann["image_id"]]
        w, h = img["width"], img["height"]

        x, y, bw, bh = ann["bbox"]
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        bw_n = bw / w
        bh_n = bh / h
        cls = ann["category_id"]

        lbl_path = out_lbl_dir / (img["file_name"].replace(".jpg", ".txt"))
        with open(lbl_path, "a", encoding="utf-8") as f:
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {bw_n:.6f} {bh_n:.6f}\n")

    print(f"[{coco_path.name}] 图片 {len(img_map)} 张 / 标注 {len(coco['annotations'])} 个")


def write_data_yaml():
    # 用正斜杠绝对路径，避免 yaml 里 Windows 反斜杠转义问题
    root = str(OUT_DIR.resolve()).replace("\\", "/")
    content = (
        "# PCB 缺陷检测数据集配置（PKU-Market-PCB）\n"
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
    )
    for i, name in enumerate(CLASS_NAMES):
        content += f"  {i}: {name}\n"
    (OUT_DIR / "data.yaml").write_text(content, encoding="utf-8")
    print("[data.yaml] 已生成")


def main():
    coco_to_yolo(COCO_DIR / "train.json", OUT_DIR / "images" / "train", OUT_DIR / "labels" / "train")
    coco_to_yolo(COCO_DIR / "val.json", OUT_DIR / "images" / "val", OUT_DIR / "labels" / "val")
    write_data_yaml()
    print("数据集准备完成!")


if __name__ == "__main__":
    main()

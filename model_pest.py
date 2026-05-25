import os
import torch
from PIL import Image
from torchvision.transforms import v2 as T

# =========================
# 固定配置
# =========================
MODEL_PATH = "model/leaf_pest_detector_final.pth"
CLASSES_FILE = "model/classes.txt"
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 安全导入 get_model
# =========================
try:
    from model.model import get_model
except Exception as e:
    get_model = None
    print("虫害模型结构文件导入失败：", e)


# =========================
# 图像预处理
# =========================
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])


_model = None
_classes = None


def load_classes(classes_file):
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"找不到类别文件：{classes_file}")

    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    return classes


def load_model():
    global _model, _classes

    if _model is not None:
        return _model, _classes

    if get_model is None:
        raise RuntimeError("没有成功导入 get_model，请检查 model/model.py")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到虫害模型文件：{MODEL_PATH}")

    if not os.path.exists(CLASSES_FILE):
        raise FileNotFoundError(f"找不到类别文件：{CLASSES_FILE}")

    _classes = load_classes(CLASSES_FILE)

    print("正在加载虫害检测模型...")

    model = get_model(len(_classes))

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    model.eval()

    _model = model

    print("虫害检测模型加载完成")

    return _model, _classes


def unavailable_result(error_message):
    """
    虫害模型不可用时，返回一个正常 JSON，
    避免前端整体分析失败。
    """
    return {
        "pest_detected": False,
        "pest_label": "虫害模型暂不可用",
        "confidence": 0.0,
        "boxes": [],
        "error": error_message
    }


def predict_pest(image: Image.Image):
    """
    给 app.py 调用的虫害检测函数
    """

    try:
        model, classes = load_model()

        image = image.convert("RGB")

        image_tensor = transform(image).to(DEVICE)

        with torch.no_grad():
            prediction = model([image_tensor])[0]

        boxes_tensor = prediction["boxes"]
        labels_tensor = prediction["labels"]
        scores_tensor = prediction["scores"]

        boxes = []

        detected = False
        best_label = "无明显虫害"
        best_confidence = 0.0

        for box, label, score in zip(
            boxes_tensor,
            labels_tensor,
            scores_tensor
        ):
            score_value = float(score)

            if score_value < THRESHOLD:
                continue

            detected = True

            x1, y1, x2, y2 = box.int().tolist()

            label_id = int(label.item())

            # 有些检测模型 label 从 1 开始，classes 从 0 开始
            if label_id >= len(classes) and label_id - 1 < len(classes):
                class_name = classes[label_id - 1]
            elif 0 <= label_id < len(classes):
                class_name = classes[label_id]
            else:
                class_name = f"unknown_{label_id}"

            item = {
                "label": class_name,
                "confidence": score_value,
                "box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            }

            boxes.append(item)

            if score_value > best_confidence:
                best_confidence = score_value
                best_label = class_name

        return {
            "pest_detected": detected,
            "pest_label": best_label,
            "confidence": best_confidence,
            "boxes": boxes
        }

    except Exception as e:
        print("虫害模型运行失败：", e)
        return unavailable_result(str(e))
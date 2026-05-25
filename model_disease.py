import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# =========================
# 固定配置
# =========================
MODEL_PATH = "model/LeavesDiseases_model.pth"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "Anthracnose(炭疽病)",
    "Cordana(科达纳叶斑病)",
    "Healthy(健康)",
    "PowderyMildew(白粉病)",
    "RedRot(红腐病)",
    "Rust(锈病)",
    "Scorch(叶焦病)",
    "SootyMould(煤污病)",
    "Yellowing(黄化病)"
]


# =========================
# 图片预处理
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.45, 0.406],
        [0.224, 0.224, 0.225]
    )
])


# =========================
# 全局模型缓存
# =========================
_model = None


def load_model():
    global _model

    if _model is not None:
        return _model

    print("正在加载病害识别模型...")

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASSES)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)

    model.eval()

    _model = model

    print("病害模型加载完成")

    return _model


# =========================
# Flask 调用入口
# =========================
def predict_disease(image: Image.Image):

    model = load_model()

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        prob = torch.softmax(
            output,
            dim=1
        )

        top5_prob, top5_idx = torch.topk(
            prob,
            5
        )

    top5 = []

    for idx, p in zip(
        top5_idx[0],
        top5_prob[0]
    ):
        top5.append({
            "label": CLASSES[int(idx)],
            "confidence": float(p)
        })

    best = top5[0]

    return {
        "disease": best["label"],
        "confidence": best["confidence"],
        "top5": top5
    }
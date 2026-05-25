# =========================
# 统一模型入口
# app.py 只调用这里
# =========================

from model_species import predict_species
from model_disease import predict_disease
from model_pest import predict_pest


# =========================
# 植物种类识别
# =========================
def predict_species_interface(image):
    """
    输入：
        PIL.Image

    输出：
        {
            "species": "...",
            "confidence": 0.95
        }
    """

    result = predict_species(image)

    return result


# =========================
# 病害识别
# =========================
def predict_disease_interface(image):
    """
    输入：
        PIL.Image

    输出：
        {
            "disease":"Rust(锈病)",
            "confidence":0.88,
            "top5":[]
        }
    """

    result = predict_disease(image)

    return result


# =========================
# 虫害识别
# =========================
def predict_pest_interface(image):
    """
    输入：
        PIL.Image

    输出：
        {
            "pest_detected": True,
            "pest_label":"蚜虫",
            "confidence":0.92,
            "boxes":[]
        }
    """

    result = predict_pest(image)

    return result
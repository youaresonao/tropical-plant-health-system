from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import traceback

# 从统一入口调用三个模型
from model_loader import (
    predict_species_interface,
    predict_disease_interface,
    predict_pest_interface
)

# =========================
# Flask 初始化
# =========================
app = Flask(
    __name__,
    static_folder="static"
)

CORS(app)

# 限制上传大小：10MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# =========================
# 首页
# =========================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# =========================
# 读取上传图片
# =========================
def read_image_from_request():
    """
    从前端 FormData 中读取 file 字段
    转成 PIL Image
    """

    if "file" not in request.files:
        raise ValueError("没有接收到图片")

    file = request.files["file"]

    if file.filename == "":
        raise ValueError("没有选择文件")

    image = Image.open(file.stream).convert("RGB")

    return image


# =========================
# 植物种类识别接口
# =========================
@app.route("/predict_species", methods=["POST"])
def predict_species():

    try:
        image = read_image_from_request()

        result = predict_species_interface(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# 病害识别接口
# =========================
@app.route("/predict_disease", methods=["POST"])
def predict_disease():

    try:
        image = read_image_from_request()

        result = predict_disease_interface(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# 虫害识别接口
# =========================
@app.route("/predict_pest", methods=["POST"])
def predict_pest():

    try:
        image = read_image_from_request()

        result = predict_pest_interface(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# 启动
# =========================
if __name__ == "__main__":

    print("=" * 60)
    print("植物健康评估系统已启动")
    print("浏览器打开：http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True
    )
# Tropical Plant Health System

基于深度学习的热带植物健康评估系统。

上传植物叶片图片后，系统自动完成：

- 植物种类识别
- 叶片病变识别
- 虫害检测
- 自动生成植物健康评估报告

---

## 项目结构

```txt
project/
├── app.py
├── model_species.py
├── model_disease.py
├── model_pest.py
├── model_loader.py
├── model/
│   ├── model.py
│   └── classes.txt
├── static/
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 功能介绍

### 1. 植物种类识别

识别热带植物种类：

例如：

- Anthurium clarinervium
- Monstera deliciosa
- Philodendron

---

### 2. 叶片病变识别

支持识别：

- Healthy
- Scorch
- Rust
- Powdery Mildew
- Yellowing
- Anthracnose

---

### 3. 虫害检测

支持：

- 是否检测到虫害
- 虫害类别
- 目标框定位

---

### 4. 健康评估报告

根据：

- 植物种类
- 病害
- 虫害

自动生成综合养护建议。

---

## 模型文件说明

由于模型文件较大，未直接上传到 GitHub。

请先下载模型文件，然后放入项目的 `model/` 目录中。

百度网盘下载地址：

```txt
链接: https://pan.baidu.com/s/1KTzq5EReOAst3EGggvsckg?pwd=wgxw
提取码: wgxw
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 启动

```bash
python app.py
```

浏览器打开：

```txt
http://127.0.0.1:5000
```

---

## 技术栈

- Python
- Flask
- PyTorch
- OpenCLIP
- HTML
- CSS
- JavaScript

---

## 项目说明

课程实验项目。

识别结果用于辅助判断。

最终请结合人工观察判断植物健康状态。

---

## 作者

刘博洋 李世隆 傅鸿博 杨竣胜 刘子毅

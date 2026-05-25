const API_BASE = "http://127.0.0.1:5000";

const imageInput = document.getElementById("imageInput");
const dropArea = document.getElementById("dropArea");
const previewWrap = document.getElementById("previewWrap");
const previewImage = document.getElementById("previewImage");
const clearBtn = document.getElementById("clearBtn");
const analyzeBtn = document.getElementById("analyzeBtn");

const loading = document.getElementById("loading");
const emptyState = document.getElementById("emptyState");
const results = document.getElementById("results");
const reportCard = document.getElementById("reportCard");
const reportContent = document.getElementById("reportContent");

const speciesName = document.getElementById("speciesName");
const speciesConfidence = document.getElementById("speciesConfidence");
const diseaseName = document.getElementById("diseaseName");
const diseaseConfidence = document.getElementById("diseaseConfidence");
const pestName = document.getElementById("pestName");
const pestConfidence = document.getElementById("pestConfidence");
const diseaseTopList = document.getElementById("diseaseTopList");

const progressBox = document.getElementById("progressBox");
const progressText = document.getElementById("progressText");
const progressPercent = document.getElementById("progressPercent");
const progressFill = document.getElementById("progressFill");
const stepSpecies = document.getElementById("stepSpecies");
const stepDisease = document.getElementById("stepDisease");
const stepPest = document.getElementById("stepPest");

let selectedFile = null;

imageInput.addEventListener("change", (e) => {
  handleFile(e.target.files[0]);
});

clearBtn.addEventListener("click", () => {
  selectedFile = null;
  imageInput.value = "";
  previewImage.src = "";
  previewWrap.classList.add("hidden");
  analyzeBtn.disabled = true;
  resetResult();
  resetProgress();
});

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("dragover");
  handleFile(e.dataTransfer.files[0]);
});

analyzeBtn.addEventListener("click", analyzeImage);

function handleFile(file) {
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    alert("请选择图片文件！");
    return;
  }

  selectedFile = file;
  previewImage.src = URL.createObjectURL(file);
  previewWrap.classList.remove("hidden");
  analyzeBtn.disabled = false;
  resetResult();
  resetProgress();
}

async function analyzeImage() {
  if (!selectedFile) {
    alert("请先上传图片！");
    return;
  }

  setLoading(true);
  startProgress();

  try {
    setStepActive(stepSpecies, "正在识别植物种类...");
    const speciesRes = await postImage("/predict_species", selectedFile);
    setStepDone(stepSpecies);
    updateProgress(35, "植物种类识别完成");

    setStepActive(stepDisease, "正在识别叶片病变...");
    const diseaseRes = await postImage("/predict_disease", selectedFile);
    setStepDone(stepDisease);
    updateProgress(70, "叶片病变识别完成");

    setStepActive(stepPest, "正在检测虫害...");
    const pestRes = await postImage("/predict_pest", selectedFile);
    setStepDone(stepPest);
    updateProgress(100, "全部模型分析完成");

    renderResults(speciesRes, diseaseRes, pestRes);
    generateReport(speciesRes, diseaseRes, pestRes);
  } catch (error) {
    console.error(error);
    progressText.textContent = "分析失败，请检查后端接口";
    alert("分析失败：请确认 app.py 已启动，并且接口地址正确。");
  } finally {
    setLoading(false);
  }
}

async function postImage(apiPath, file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}${apiPath}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`${apiPath} 请求失败`);
  }

  return await response.json();
}

function renderResults(species, disease, pest) {
  emptyState.classList.add("hidden");
  results.classList.remove("hidden");

  const speciesResult = normalizeSpecies(species);
  const diseaseResult = normalizeDisease(disease);
  const pestResult = normalizePest(pest);

  speciesName.textContent = speciesResult.name;
  speciesConfidence.textContent = `置信度：${formatPercent(speciesResult.confidence)}`;

  diseaseName.textContent = diseaseResult.name;
  diseaseConfidence.textContent = `置信度：${formatPercent(diseaseResult.confidence)}`;

  pestName.textContent = pestResult.name;
  pestConfidence.textContent = pestResult.detected ? `置信度：${formatPercent(pestResult.confidence)}` : "未检测到明显虫害";

  diseaseTopList.innerHTML = "";
  diseaseResult.topList.forEach((item, index) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span>${index + 1}. ${item.label}</span>
      <strong>${formatPercent(item.confidence)}</strong>
    `;
    diseaseTopList.appendChild(li);
  });
}
function generateReport(species, disease, pest) {
  const s = normalizeSpecies(species);
  const d = normalizeDisease(disease);
  const p = normalizePest(pest);

  const assessment = buildSmartHealthAssessment(s, d, p);

  reportCard.classList.remove("hidden");

  reportContent.innerHTML = `
    <div class="health-level ${assessment.levelClass}">
      <span>${assessment.icon}</span>
      <div>
        <h3>${assessment.levelTitle}</h3>
        <p>${assessment.summary}</p>
      </div>
    </div>

    <div class="report-section">
      <h3>一、识别对象</h3>
      <p>系统识别该植物可能为 <b>${s.name}</b>，植物种类识别置信度为 <b>${formatPercent(s.confidence)}</b>。</p>
      <p>${assessment.speciesComment}</p>
    </div>

    <div class="report-section">
      <h3>二、叶片病变分析</h3>
      <p>病变模型识别结果为 <b>${d.name}</b>，置信度为 <b>${formatPercent(d.confidence)}</b>。</p>
      <p><b>结果解释：</b>${assessment.diseaseExplanation}</p>
      <p><b>可信程度：</b>${assessment.diseaseReliability}</p>
    </div>

    <div class="report-section">
      <h3>三、虫害检测分析</h3>
      <p>${assessment.pestDescription}</p>
      <p><b>虫害风险说明：</b>${assessment.pestRisk}</p>
    </div>

    <div class="report-section">
      <h3>四、综合风险判断</h3>
      <p>${assessment.riskReason}</p>
    </div>

    <div class="report-section">
      <h3>五、养护与处理建议</h3>
      <ul class="advice-list">
        ${assessment.suggestions.map(item => `<li>${item}</li>`).join("")}
      </ul>
    </div>

    <div class="report-section">
      <h3>六、说明</h3>
      <p>本报告由植物种类识别模型、叶片病变识别模型和虫害检测模型的输出结果综合生成，仅作为辅助判断。实际养护或防治时，建议结合人工观察、叶片背面检查和连续多日状态变化进行确认。</p>
    </div>
  `;
}


function buildSmartHealthAssessment(species, disease, pest) {
  const diseaseName = String(disease.name || "").toLowerCase();
  const diseaseConf = Number(disease.confidence || 0);
  const pestConf = Number(pest.confidence || 0);

  const isHealthy =
    diseaseName.includes("healthy") ||
    diseaseName.includes("健康");

  const diseaseInfo = getDiseaseKnowledge(disease.name);
  const reliability = getConfidenceText(diseaseConf);

  let riskScore = 0;

  if (!isHealthy) riskScore += 45;
  if (diseaseConf >= 0.7 && !isHealthy) riskScore += 20;
  if (diseaseConf < 0.45 && !isHealthy) riskScore -= 10;
  if (pest.detected) riskScore += 30;
  if (pest.detected && pestConf >= 0.7) riskScore += 10;

  riskScore = Math.max(0, Math.min(100, riskScore));

  let levelClass = "good";
  let icon = "✅";
  let levelTitle = "健康状态良好";
  let summary = "当前图片中未发现明显病变或虫害特征。";

  if (riskScore >= 70) {
    levelClass = "danger";
    icon = "⚠️";
    levelTitle = "较高风险";
    summary = "系统检测到较明显的叶片异常或虫害风险，建议尽快处理并持续观察。";
  } else if (riskScore >= 35) {
    levelClass = "warning";
    icon = "🔍";
    levelTitle = "中等风险";
    summary = "系统检测到一定程度的叶片异常，建议结合人工观察进一步确认。";
  } else if (!isHealthy) {
    levelClass = "warning";
    icon = "🌿";
    levelTitle = "轻度风险";
    summary = "模型识别到轻微异常，但风险程度相对较低。";
  }

  const pestDescription = pest.detected
    ? `虫害模型检测到疑似虫害：<b>${pest.name}</b>，置信度为 <b>${formatPercent(pest.confidence)}</b>。`
    : `虫害模型暂未检测到明显虫害。`;

  const pestRisk = pest.detected
    ? "检测到虫害目标时，应重点检查叶背、嫩芽、茎节等位置，因为许多虫害早期集中在这些区域。"
    : "未检测到虫害不代表完全没有虫害，仍建议人工查看叶背、新叶和叶柄位置。";

  const speciesComment = getSpeciesComment(species.name);

  const riskReason = makeRiskReason(isHealthy, disease, pest, riskScore);

  const suggestions = buildSuggestions(
  isHealthy,
  disease.name,
  pest.detected,
  species.name,
  pest.name
  );

  return {
    levelClass,
    icon,
    levelTitle,
    summary,
    speciesComment,
    diseaseExplanation: diseaseInfo.explanation,
    diseaseReliability: reliability,
    pestDescription,
    pestRisk,
    riskReason,
    suggestions
  };
}


function getDiseaseKnowledge(name) {
  if (!name) {
    return {
      explanation: "模型未返回明确病害类别，建议重新上传清晰叶片图片进行识别。"
    };
  }

  if (name.includes("Healthy") || name.includes("健康")) {
    return {
      explanation: "模型未识别到明显病斑、霉层、锈斑、焦枯或异常黄化等特征，叶片整体状态较正常。"
    };
  }

  if (name.includes("Anthracnose") || name.includes("炭疽")) {
    return {
      explanation: "炭疽病通常表现为叶片出现褐色或黑褐色病斑，病斑可能逐渐扩大，潮湿环境下更容易扩散。"
    };
  }

  if (name.includes("Cordana") || name.includes("科达纳")) {
    return {
      explanation: "科达纳叶斑病通常表现为叶片上出现边界较明显的斑块，可能与真菌感染、湿度较高和通风不足有关。"
    };
  }

  if (name.includes("Powdery") || name.includes("白粉")) {
    return {
      explanation: "白粉病常表现为叶片表面出现白色粉状物，通常与通风不良、湿度变化和植株过密有关。"
    };
  }

  if (name.includes("RedRot") || name.includes("红腐")) {
    return {
      explanation: "红腐病一般会造成组织变色、腐烂或坏死，若持续发展可能影响植株整体生长。"
    };
  }

  if (name.includes("Rust") || name.includes("锈病")) {
    return {
      explanation: "锈病常表现为叶片出现黄褐色、橙褐色或铁锈色斑点，通常与高湿环境和叶面长期潮湿有关。"
    };
  }

  if (name.includes("Scorch") || name.includes("叶焦")) {
    return {
      explanation: "叶焦病常表现为叶尖、叶缘干枯或褐化，可能与强光暴晒、缺水、肥害、空气湿度不足或根系吸水异常有关。"
    };
  }

  if (name.includes("Sooty") || name.includes("煤污")) {
    return {
      explanation: "煤污病通常表现为叶片表面出现黑色煤灰状覆盖物，常与蚜虫、介壳虫等害虫分泌蜜露有关。"
    };
  }

  if (name.includes("Yellowing") || name.includes("黄化")) {
    return {
      explanation: "黄化通常表现为叶片发黄，可能与缺素、浇水不当、光照不足、根系问题或环境变化有关。"
    };
  }

  return {
    explanation: "模型检测到叶片颜色、纹理或形态存在异常，建议结合人工观察进一步确认具体原因。"
  };
}


function getConfidenceText(confidence) {
  const value = Number(confidence || 0);

  if (value >= 0.8) {
    return "模型置信度较高，当前识别结果具有较强参考价值。";
  }

  if (value >= 0.55) {
    return "模型置信度中等，结果可以作为参考，但建议结合人工观察复核。";
  }

  if (value > 0) {
    return "模型置信度偏低，说明图片特征可能不够明显，建议更换更清晰、光照更均匀的叶片图片重新检测。";
  }

  return "当前模型未提供有效置信度，建议仅作为辅助信息参考。";
}


function getSpeciesComment(name) {
  const lower = String(name || "").toLowerCase();

  if (lower.includes("monstera") || name.includes("龟背竹")) {
    return "龟背竹类植物叶片较大，常见问题包括叶缘焦枯、黄化和湿度不足导致的叶面损伤。";
  }

  if (lower.includes("anthurium") || name.includes("花烛") || name.includes("红掌")) {
    return "花烛类植物对湿度和通风较敏感，叶片出现斑点或焦边时，应重点检查浇水、光照和空气湿度。";
  }

  if (lower.includes("alocasia") || name.includes("海芋")) {
    return "海芋类植物叶片较薄，对强光、低温和浇水变化较敏感，容易出现叶缘焦枯或黄化。";
  }

  if (lower.includes("ficus") || name.includes("榕")) {
    return "榕属植物对环境变化较敏感，若出现黄叶或落叶，应关注光照变化、浇水频率和根系状态。";
  }

  if (lower.includes("mangifera") || name.includes("芒果")) {
    return "芒果叶片常见问题包括炭疽、叶斑、虫害取食和叶缘焦枯，需要结合叶片斑点形态进一步判断。";
  }

  return "不同植物对光照、湿度和病虫害的敏感程度不同，后续建议结合该植物的实际养护环境进行判断。";
}


function makeRiskReason(isHealthy, disease, pest, riskScore) {
  if (isHealthy && !pest.detected) {
    return "病害模型结果为健康，虫害模型也未检测到明显虫害，因此综合判断当前风险较低。";
  }

  if (!isHealthy && pest.detected) {
    return `病害模型识别到 ${disease.name}，同时虫害模型检测到 ${pest.name}，病害和虫害风险叠加，因此综合风险较高。`;
  }

  if (!isHealthy) {
    return `病害模型识别到 ${disease.name}，虽然当前未检测到明显虫害，但叶片已经存在异常特征，因此综合风险评分约为 ${riskScore} / 100。`;
  }

  if (pest.detected) {
    return `病害模型未识别到明显病变，但虫害模型检测到 ${pest.name}，说明可能处于虫害早期或局部受害阶段。`;
  }

  return "当前模型结果未显示明显异常，但仍建议定期观察植株状态。";
}


function buildSuggestions(isHealthy, diseaseName, hasPest, plantName = "", pestName = "") {
  const smartSuggestions = buildPlantSpecificSuggestions(
    plantName,
    diseaseName,
    pestName,
    hasPest
  );

  if (smartSuggestions.length > 0) {
    return smartSuggestions;
  }

  return [
    "建议重新上传更清晰的叶片图片，或从叶片正面、背面分别拍摄进行复核。",
    "结合实际养护环境判断光照、浇水、通风和湿度是否存在异常。",
    "持续观察 3 到 7 天，记录叶片病斑是否扩大或虫害是否增加。"
  ];
}

function normalizeSpecies(data) {
  return {
    name: data.species || data.plant_name || data.predicted_label || data.label || "未知植物",
    confidence: data.confidence || data.probability || data.score || 0,
  };
}

function normalizeDisease(data) {
  let topList = [];

  if (Array.isArray(data.top5)) {
    topList = data.top5.map((item) => ({
      label: item.label || item.name || item.disease || "未知病害",
      confidence: item.confidence || item.probability || item.score || 0,
    }));
  } else if (Array.isArray(data.results)) {
    topList = data.results.map((item) => ({
      label: item.label || item.name || item.disease || "未知病害",
      confidence: item.confidence || item.probability || item.score || 0,
    }));
  }

  const first = topList[0];

  return {
    name: data.disease || data.disease_type || data.label || data.predicted_label || first?.label || "未知病变",
    confidence: data.confidence || data.probability || data.score || first?.confidence || 0,
    topList: topList.length > 0 ? topList : [{
      label: data.disease || data.disease_type || data.label || "暂无 Top 结果",
      confidence: data.confidence || data.probability || data.score || 0,
    }],
  };
}

function normalizePest(data) {
  const detected = data.pest_detected || data.detected || data.has_pest || false;

  return {
    detected,
    name: data.pest_label || data.pest || data.label || (detected ? "疑似虫害" : "无明显虫害"),
    confidence: data.confidence || data.probability || data.score || 0,
  };
}


function formatPercent(value) {
  const num = Number(value);
  if (Number.isNaN(num)) return "--";
  return `${(num * 100).toFixed(1)}%`;
}

function setLoading(isLoading) {
  loading.classList.toggle("hidden", !isLoading);
  analyzeBtn.disabled = isLoading || !selectedFile;
  analyzeBtn.textContent = isLoading ? "分析中..." : "开始智能分析";
}

function startProgress() {
  progressBox.classList.remove("hidden");
  updateProgress(8, "准备上传图片并调用模型...");
  stepSpecies.className = "";
  stepDisease.className = "";
  stepPest.className = "";
}

function updateProgress(percent, text) {
  progressFill.style.width = `${percent}%`;
  progressPercent.textContent = `${Math.round(percent)}%`;
  progressText.textContent = text;
}

function setStepActive(step, text) {
  step.classList.add("active");
  progressText.textContent = text;
}

function setStepDone(step) {
  step.classList.remove("active");
  step.classList.add("done");
}

function resetProgress() {
  progressBox.classList.add("hidden");
  updateProgress(0, "准备分析...");
  stepSpecies.className = "";
  stepDisease.className = "";
  stepPest.className = "";
}

function resetResult() {
  emptyState.classList.remove("hidden");
  results.classList.add("hidden");
  reportCard.classList.add("hidden");
}

function buildPlantSpecificSuggestions(plantName, diseaseName, pestName, hasPest) {
  const plant = String(plantName || "");
  const disease = String(diseaseName || "");
  const pest = String(pestName || "");

  const suggestions = [];

  const plantType = getPlantType(plant);

  // 1. 先根据植物类型给基础建议
  if (plantType === "aroid") {
    suggestions.push("该植物属于天南星科观叶植物，建议放在明亮散射光环境，避免强光直射导致叶片焦边。");
    suggestions.push("保持较高空气湿度，但不要让盆土长期积水，否则容易诱发根系和叶片问题。");
  }

  if (plantType === "succulent") {
    suggestions.push("该植物偏多肉类，建议减少浇水频率，保持土壤干透后再浇，避免高湿环境导致腐烂。");
    suggestions.push("病害处理时应优先改善通风和控水，不建议频繁喷雾。");
  }

  if (plantType === "fern") {
    suggestions.push("该植物偏蕨类，喜欢较高湿度和散射光，避免空气过干导致叶缘干枯。");
    suggestions.push("保持基质微湿，但需要避免盆底积水。");
  }

  if (plantType === "flowering") {
    suggestions.push("该植物属于开花植物，病害处理时需要兼顾花期状态，避免在强光或高温时喷药。");
    suggestions.push("若叶片异常同时伴随花苞脱落，应重点检查浇水和通风条件。");
  }

  if (plantType === "fruit") {
    suggestions.push("该植物属于果树类或热带经济植物，叶片病斑可能影响后续生长，需要更重视早期隔离和病叶清理。");
  }

  // 2. 根据病害类型叠加建议
  if (disease.includes("Healthy") || disease.includes("健康")) {
    suggestions.push("当前病害模型判断为健康，建议继续保持现有养护方式，并定期检查叶背和新叶。");
  }

  else if (disease.includes("Scorch") || disease.includes("叶焦")) {
    if (plantType === "aroid" || plantType === "fern") {
      suggestions.push("叶焦对这类植物常与空气湿度不足、强光直射或浇水波动有关，建议提高空气湿度并改为散射光养护。");
    } else if (plantType === "succulent") {
      suggestions.push("多肉类出现叶焦时，优先检查是否突然暴晒或盆土过干，不建议通过大量浇水快速补救。");
    } else {
      suggestions.push("叶焦通常与强光、缺水、肥害或环境突变有关，建议先调整光照和浇水频率。");
    }
  }

  else if (disease.includes("Rust") || disease.includes("锈病")) {
    suggestions.push("锈病通常与高湿、通风不足有关，建议减少叶面喷水，清理病叶并增强空气流通。");

    if (plantType === "flowering") {
      suggestions.push("开花植物出现锈病时，建议避开花朵喷洒药剂，以免影响花期观赏。");
    }
  }

  else if (disease.includes("Powdery") || disease.includes("白粉")) {
    suggestions.push("白粉病多与通风不良和植株过密有关，建议增加植株间距，减少叶面长期潮湿。");

    if (plantType === "succulent") {
      suggestions.push("多肉类发生白粉类问题时，不建议频繁喷水，应优先加强通风和降低环境湿度。");
    }
  }

  else if (disease.includes("Anthracnose") || disease.includes("炭疽")) {
    suggestions.push("炭疽病有扩散风险，建议剪除明显病斑叶片，并将植株与健康植株隔离观察。");

    if (plant.includes("Mangifera") || plant.includes("芒果")) {
      suggestions.push("芒果类植物较常见炭疽病，若病斑持续扩大，应重点控制湿度并清除落叶残枝。");
    }
  }

  else if (disease.includes("Sooty") || disease.includes("煤污")) {
    suggestions.push("煤污病常与蚜虫、介壳虫等害虫分泌蜜露有关，建议同时检查虫害来源。");
  }

  else if (disease.includes("Yellowing") || disease.includes("黄化")) {
    if (plantType === "aroid") {
      suggestions.push("天南星科植物黄化常与积水、根系缺氧或光照不足有关，建议检查盆土湿度和根系状态。");
    } else if (plantType === "succulent") {
      suggestions.push("多肉类黄化可能与浇水过多、光照不足或根系腐烂有关，应优先控水并增加光照。");
    } else {
      suggestions.push("黄化可能与浇水、光照、营养或根系状态有关，建议结合叶片位置和新老叶变化判断。");
    }
  }

  // 3. 根据虫害叠加建议
  if (hasPest) {
    if (pest.includes("aphid") || pest.includes("蚜")) {
      suggestions.push("检测到疑似蚜虫时，应重点检查嫩芽、叶背和新叶，可先用清水冲洗或人工清除。");
    } else if (pest.includes("spider") || pest.includes("螨") || pest.includes("红蜘蛛")) {
      suggestions.push("疑似红蜘蛛或螨类时，通常与空气干燥有关，建议提高湿度并重点检查叶背细小虫点和蛛丝。");
    } else if (pest.includes("scale") || pest.includes("介壳")) {
      suggestions.push("疑似介壳虫时，建议用棉签蘸酒精擦除虫体，并隔离植株持续观察。");
    } else {
      suggestions.push("检测到疑似虫害时，建议重点检查叶背、嫩芽、叶柄和茎节位置，必要时人工复核。");
    }

    if (plantType === "succulent") {
      suggestions.push("多肉类发生虫害时，处理后应保持伤口干燥，避免立刻大量喷水。");
    }

    if (plantType === "fern") {
      suggestions.push("蕨类植物叶片细密，虫害容易隐藏，建议分层翻看叶片背面。");
    }
  }

  // 4. 去重
  return [...new Set(suggestions)];
}

function getPlantType(plantName) {
  const name = String(plantName || "").toLowerCase();

  if (
    name.includes("monstera") ||
    name.includes("philodendron") ||
    name.includes("epipremnum") ||
    name.includes("scindapsus") ||
    name.includes("anthurium") ||
    name.includes("spathiphyllum") ||
    name.includes("alocasia") ||
    name.includes("syngonium") ||
    name.includes("dieffenbachia") ||
    name.includes("aglaonema") ||
    plantName.includes("龟背竹") ||
    plantName.includes("绿萝") ||
    plantName.includes("蔓绿绒") ||
    plantName.includes("花烛") ||
    plantName.includes("红掌") ||
    plantName.includes("白掌") ||
    plantName.includes("海芋") ||
    plantName.includes("合果芋") ||
    plantName.includes("万年青")
  ) {
    return "aroid";
  }

  if (
    name.includes("echeveria") ||
    name.includes("haworthia") ||
    name.includes("aloe") ||
    name.includes("sedum") ||
    name.includes("crassula") ||
    name.includes("kalanchoe") ||
    name.includes("agave") ||
    plantName.includes("多肉") ||
    plantName.includes("芦荟") ||
    plantName.includes("龙舌兰") ||
    plantName.includes("玉露")
  ) {
    return "succulent";
  }

  if (
    name.includes("nephrolepis") ||
    name.includes("adiantum") ||
    name.includes("asplenium") ||
    name.includes("platycerium") ||
    plantName.includes("蕨") ||
    plantName.includes("鸟巢") ||
    plantName.includes("鹿角")
  ) {
    return "fern";
  }

  if (
    name.includes("rosa") ||
    name.includes("gardenia") ||
    name.includes("jasminum") ||
    name.includes("hibiscus") ||
    name.includes("bougainvillea") ||
    name.includes("hydrangea") ||
    plantName.includes("月季") ||
    plantName.includes("栀子") ||
    plantName.includes("茉莉") ||
    plantName.includes("朱槿") ||
    plantName.includes("三角梅") ||
    plantName.includes("绣球")
  ) {
    return "flowering";
  }

  if (
    name.includes("mangifera") ||
    name.includes("citrus") ||
    name.includes("punica") ||
    plantName.includes("芒果") ||
    plantName.includes("柠檬") ||
    plantName.includes("橘") ||
    plantName.includes("金桔") ||
    plantName.includes("石榴")
  ) {
    return "fruit";
  }

  return "general";
}
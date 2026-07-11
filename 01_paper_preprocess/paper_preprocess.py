# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONCURRENCY = 50
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAX_RETRIES = 2
DEFAULT_PDF_RENDER_DPI = 300
IMAGE_MIME_TYPE = "image/png"
IMAGE_FORMAT = "png"


EXTRACTION_PROMPT = """
你是一个专业的涂层文献信息提取助手。你的任务是从提供的学术文献中提取与涂层制备、微观结构和性能相关的关键信息，并严格按照给定的 JSON 格式输出。提取结果必须是一个合法的 JSON 对象，不包含任何额外的文本、解释或 Markdown 标记。

## 输入
用户将提供一篇或多篇涂层领域的学术文献文本（可能包含摘要、全文段落等）。

## 输出要求
1. 输出必须是一个单一、完整的 JSON 对象。
2. 所有字段都必须出现，若无对应信息则用 `null` 表示；空数组用 `[]` 表示。
3. 数值型字段直接填入数字，不需要带单位（字段名中已注明单位，且默认使用指定单位）。若文献中的单位与字段单位不一致，请换算为指定单位后再填入数值。
4. 保持样品的关联性：同一个涂层样品在文献中会有唯一标识（如编号 S1、C1 等），请在 `样品/涂层编号` 字段中使用原文给出的编号，并在不同部分（喷涂工艺、微观组织、力学性能等）中保持一致。
5. 若文献中存在多种粉末或多种涂层，请分别创建数组中的多个对象，并用 `粉末编号` 或 `样品/涂层编号` 加以区分。

## JSON 结构模板及提取规则

请严格按照下面的结构输出，每个字段的含义和提取规则如下：

```json
{
  "文献元数据": {
    "标题": "文献标题",
    "DOI": "DOI号",
    "发表年份": 2023,
    "期刊": "期刊名称",
    "作者": ["作者1", "作者2"],
    "作者单位": ["单位1", "单位2"],
    "关键词": ["关键词1", "关键词2"]
  },
  "基体信息": {
    "基体材料": "基体材料牌号或名称（如 316L不锈钢）",
    "预处理工艺": "例如喷砂、丙酮清洗、预热等描述",
    "表面粗糙度Ra(μm)": "喷砂后的表面粗糙度Ra值（数值）",
    "基体尺寸": "长×宽×厚 (mm) 或描述性文字"
  },
  "粉末属性": [
    {
      "粉末编号": "文献中对该粉末的编号，如 P1",
      "粉末名称/牌号": "商品名或牌号，如 Metco 204NS",
      "粉末制备方法": "气雾化、水雾化、机械合金化、烧结破碎等",
      "粒径分布": {
        "粒径范围(μm)": "如 15-45，填文本",
        "D10(μm)": null,
        "D50(μm)": null,
        "D90(μm)": null
      },
      "粉末形貌": "球形、不规则、空心球等",
      "松装密度(g/cm³)": null,
      "流动性(s/50g)": null,
      "增强相类型": "如 WC、Cr₃C₂、Al₂O₃ 等，若为纯金属/合金则填 null",
      "润滑相类型": "如 MoS₂、石墨、h-BN 等，若未添加则填 null",
      "基体类型/种类": "该粉末主要适用的基体材料类别（如 钛合金、不锈钢），或文献中明确使用的基体种类",
      "化学成分": [
        {
          "元素": "元素符号，如 Ni",
          "质量分数(wt%)": 75.0
        }
      ]
    }
  ],
  "喷涂工艺参数": [
    {
      "样品/涂层编号": "涂层样品的编号，如 S1、C1",
      "功能层类型": "粘结层、工作层、封孔层等，若无明确分层则填 工作层",
      "喷涂技术类型": "从以下列表中选择：APS, HVOF, HVAF, 电弧喷涂, 冷喷涂, 火焰喷涂, 爆炸喷涂, VPS/LPPS, SPS/SPPS, PS-PVD",
      "工艺参数": {
        // 该对象内部的字段完全取决于“喷涂技术类型”的选择，具体见下文的工艺参数对照表。
        // 只填充与选定技术相关的字段，不要添加其他技术的字段。
        // 特别注意：以下三种最常用工艺必须确保提取所有列出的核心参数，其他工艺参照对照表。
      }
    }
  ],
  "微观组织结构": [
    {
      "样品/涂层编号": "与喷涂工艺中一致的编号",
      "孔隙率(%)": null,
      "涂层厚度(μm)": null,
      "相组成": "如 γ-Ni, Cr₃C₂, α-Al₂O₃，多个相用逗号分隔",
      "界面结合状态": "机械结合、冶金结合、扩散结合等",
      "组织结构": "对涂层微观形貌的整体描述，例如 层状结构、柱状晶、等轴晶、致密等特征",
      "未熔颗粒含量(%)": null,
      "氧化物含量(%)": null,
      "裂纹密度": "描述或定量数值",
      "层状结构特征": "层状结构厚度、层间结合等描述",
      "涂层密度(g/cm³)": null
    }
  ],
  "力学性能参数": [
    {
      "样品/涂层编号": "编号",
      "显微硬度": {
        "类型": "维氏、努氏等",
        "载荷(gf或N)": "如 200gf 或 1.96N",
        "测试位置": "截面/表面",
        "平均值": null,
        "标准差": null
      },
      "结合强度(MPa)": {
        "测试标准": "如 ASTM C633",
        "平均值": null,
        "断裂位置": "涂层内部/界面/胶层等"
      },
      "弹性模量(GPa)": null,
      "断裂韧性(MPa·m½)": null,
      "残余应力": {
        "类型": "拉应力/压应力",
        "数值(MPa)": null,
        "测试方法": "XRD曲率法/钻孔法"
      }
    }
  ],
  "摩擦学性能": [
    {
      "样品/涂层编号": "编号",
      "测试条件": {
        "对磨材料": "如 Al₂O₃球、GCr15钢",
        "载荷(N)": null,
        "滑动速度": "如 0.1 m/s 或 200 rpm，可带单位",
        "温度(℃)": null,
        "润滑状态": "干摩擦、油润滑等",
        "测试设备": "球-盘摩擦试验机等"
      },
      "摩擦系数": {
        "稳态平均值": null,
        "波动范围": null
      },
      "磨损率": {
        "数值": null,
        "单位": "如 mm³/Nm、mg/km"
      },
      "磨损机理": "磨粒磨损、黏着磨损、疲劳磨损等",
      "磨损产物": "磨损碎屑的形貌或成分描述，如 氧化物磨屑、层状剥落碎片等"
    }
  ],
  "腐蚀性能": [
    {
      "样品/涂层编号": "编号",
      "试验类型": "电化学腐蚀、浸泡腐蚀、盐雾腐蚀等",
      "腐蚀介质/环境": "如 3.5% NaCl溶液、1M HCl",
      "温度(℃)": null,
      "开路电位(V)": null,
      "腐蚀电流密度(A/cm²)": null,
      "点蚀电位(V)": null,
      "钝化膜电阻(Ω·cm²)": null,
      "腐蚀机理": "点蚀、晶间腐蚀、均匀腐蚀等",
      "腐蚀产物": "腐蚀后表面产物的组成或形貌",
      "拟合电路": "等效电路模型，如 R(QR) 或文字描述",
      "备注": "其他腐蚀相关数据，如氧化增重等"
    }
  ],
  "涂层组分": [
    {
      "样品/涂层编号": "编号",
      "组分类型": "元素组成 或 相组成",
      "测试方法": "EDS、XRD、XPS 等",
      "组分详情": [
        {
          "组分名称": "元素或相的名称",
          "含量": "质量分数(%) 或 体积分数(%)",
          "单位": "wt% 或 vol%"
        }
      ],
      "备注": null
    }
  ],
  "热物理性能": [
    {
      "样品/涂层编号": "编号",
      "热导率(W/mK)": {
        "测试温度(℃)": null,
        "数值": null
      },
      "热膨胀系数(10⁻⁶/K)": {
        "温度区间(℃)": null,
        "数值": null
      },
      "抗热震性能": "描述或定量"
    }
  ],
  "耐久性试验": [
    {
      "样品/涂层编号": "编号",
      "试验类型": "热循环、盐雾、高温蠕变等",
      "试验参数": "如温度范围、保持时间等",
      "失效模式": "剥落、开裂等",
      "寿命": "循环次数或小时"
    }
  ]
}
```

## 喷涂工艺参数对照表（核心三种必选，其余参照执行）

在填充 `工艺参数` 对象时，只能包含与所选 `喷涂技术类型` 相对应的字段。**对于 APS、HVOF、电弧喷涂这三种最常用的工艺，必须严格按照下表第一列的核心参数进行提取**（这些核心参数与你给出的字段标准完全一致），同时可以补充对照表中的其他常见参数（如载气流量、步距等），但不得遗漏核心参数。

| 喷涂技术类型 | 核心参数（必须优先提取） | 其他常见参数（可选，有则提取） |
|--------------|---------------------------|-------------------------------|
| **APS** (大气等离子喷涂) | 电流(A), 电压(V), 功率(kW), 主气流量(L/min), 次气流量(L/min), 喷涂距离(mm), 送粉率(g/min), 喷枪移动速度(mm/s), 基体预热温度(℃), 后处理工艺 | 等离子气体类型及比例, 载气流量(L/min), 步距(mm), 冷却方式 |
| **HVOF** (超音速火焰喷涂) | 氧气流量(L/min), 燃料流量(L/min或L/h), 燃烧比, 喷涂距离(mm), 送粉率(g/min), 喷枪移动速度(mm/s), 基体预热温度(℃), 后处理工艺 | 燃料类型, 燃烧室压力(bar), 步距(mm), 冷却方式 |
| **电弧喷涂** | 电流(A), 电压(V), 气体压力(MPa), 喷涂距离(mm), 送丝速度(m/min), 喷枪移动速度(mm/s), 基体预热温度(℃), 后处理工艺 | 雾化气体类型, 步距(mm), 冷却方式 |
| **HVAF** | 燃料类型, 空气流量(L/min), 燃料流量(L/min), 燃烧室压力(bar), 喷涂距离(mm), 送粉率(g/min), 喷枪移动速度(mm/s), 基体预热温度(℃), 后处理工艺 | 步距(mm), 冷却方式 |
| **冷喷涂** | 工作气体类型及压力(MPa), 气体预热温度(℃), 喷涂距离(mm), 送粉率(g/min), 喷枪移动速度(mm/s), 基体预热温度(℃), 后处理工艺 | 载气类型及流量, 步距(mm), 粉末预热温度(℃) |
| **火焰喷涂** | 火焰类型, 燃气类型及压力, 氧气压力(MPa), 送粉率(g/min)或送丝速度(m/min), 喷涂距离(mm), 喷枪移动速度(mm/s), 基体预热温度(℃), 后处理工艺 | 步距(mm) |
| **爆炸喷涂** | 燃气类型, 氧气比例, 爆炸频率(Hz), 喷涂距离(mm), 送粉量(g/shot), 喷枪移动速度(mm/s), 基体预热温度(℃), 后处理工艺 | 充填气体类型, 步距(mm) |
| **VPS/LPPS** | 电流(A), 电压(V), 功率(kW), 主气流量(L/min), 喷涂距离(mm), 送粉率(g/min), 喷枪移动速度(mm/s), 腔室压力(mbar或Pa), 基体预热温度(℃), 后处理工艺 | 等离子气体类型及比例, 次气流量, 载气流量, 步距(mm) |
| **SPS/SPPS** | 电流(A), 电压(V), 功率(kW), 喷涂距离(mm), 液体送料速率(mL/min), 喷枪移动速度(mm/s), 基体预热温度(℃), 后处理工艺 | 等离子气体类型, 载气流量, 液滴雾化参数, 步距(mm) |
| **PS-PVD** | 电流(A), 电压(V), 功率(kW), 喷涂距离(mm), 送粉率(g/min), 腔室压力(Pa), 喷枪移动速度(mm/s), 基体预热温度(℃), 后处理工艺 | 等离子气体类型及比例, 载气流量, 步距(mm) |

**工艺参数填写示例**（HVOF）：
```json
"工艺参数": {
  "氧气流量(L/min)": 900,
  "燃料流量(L/h)": 25,
  "燃烧比": 1.1,
  "喷涂距离(mm)": 350,
  "送粉率(g/min)": 60,
  "喷枪移动速度(mm/s)": 800,
  "基体预热温度(℃)": 150,
  "后处理工艺": "未处理",
  "燃料类型": "煤油",
  "步距(mm)": 6,
  "冷却方式": "压缩空气冷却"
}
```

## 重要注意事项
- 文献中若未明确给出样品编号，可自行分配编号（如 C1、C2），并确保前后一致。
- 一个喷涂工艺对象对应一个涂层样品的一个功能层；如果一个样品有粘结层和工作层两次喷涂，则需要拆成两个对象，分别填写功能层类型和对应的工艺参数。
- 力学性能、摩擦学性能、腐蚀性能等数据必须用与涂层样品一致的编号关联。
- **粉末属性中的“基体类型/种类”** 指该粉末在文献中涂覆的基体材料大类（如钛合金、不锈钢），不可与“基体信息”混淆；若无明确信息可填 null。
- **“涂层组分”** 用于记录涂层最终的元素或相组成（由 EDS、XRD 等分析得出），请与粉末的原始化学成分区分开。
- 只提取文献中明确报道的内容，不要推测和编造。
- 最终输出必须是一个完整的 JSON 对象，直接输出纯 JSON 字符串，不添加任何代码块标记。

现在，请根据以上规则，提取用户提供的文献中的信息。
""".strip()


class ConfigError(RuntimeError):
    pass


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def format_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got: {raw_value}") from exc
    if value < 1:
        raise ConfigError(f"{name} must be greater than or equal to 1, got: {value}")
    return value


def load_config() -> dict[str, str]:
    config = {
        "api_key": os.getenv("API_KEY", "").strip(),
        "responses_url": os.getenv("RESPONSES_URL", "").strip(),
        "model": os.getenv("MODEL", "").strip(),
    }

    missing = [
        name
        for name, value in {
            "API_KEY": config["api_key"],
            "RESPONSES_URL": config["responses_url"],
            "MODEL": config["model"],
        }.items()
        if not value
    ]
    if missing:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Please create .env from .env.example."
        )
    return config


def find_pdf_files(paper_dir: Path) -> list[Path]:
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")
    if not paper_dir.is_dir():
        raise NotADirectoryError(f"Paper path is not a directory: {paper_dir}")
    return sorted(path for path in paper_dir.rglob("*.pdf") if path.is_file())


def post_with_retries(
    url: str,
    *,
    headers: dict[str, str],
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, timeout=DEFAULT_TIMEOUT_SECONDS, **kwargs)
            if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                if attempt < DEFAULT_MAX_RETRIES:
                    time.sleep(2**attempt)
                    continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < DEFAULT_MAX_RETRIES:
                time.sleep(2**attempt)
                continue
            if getattr(exc, "response", None) is not None and exc.response is not None:
                body = exc.response.text[:1000]
                raise RuntimeError(f"API request failed: {exc}; response body: {body}") from exc
            raise RuntimeError(f"API request failed: {exc}") from exc
    raise RuntimeError(f"API request failed: {last_error}")


def render_pdf_as_image_parts(
    path: Path,
) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: PyMuPDF. Install it with 'pip install PyMuPDF'."
        ) from exc

    zoom = DEFAULT_PDF_RENDER_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    image_parts: list[dict[str, Any]] = []

    with fitz.open(path) as document:
        if document.page_count < 1:
            raise RuntimeError(f"PDF has no pages: {path}")

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_bytes = pixmap.tobytes(IMAGE_FORMAT)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            image_parts.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{IMAGE_MIME_TYPE};base64,{encoded}"
                }
            )

    return image_parts


def request_extraction(
    image_parts: list[dict[str, Any]],
    config: dict[str, str],
) -> str:
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    request_body = {
        "model": config["model"],
        "input": [
            {
                "role": "user",
                "content": [
                    *image_parts,
                    {"type": "input_text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
    }
    response = post_with_retries(
        config["responses_url"],
        headers=headers,
        json=request_body,
    )
    response_payload = response.json()
    output = response_payload.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for content_item in content:
                if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                    return content_item["text"].strip()

    if isinstance(response_payload.get("output_text"), str):
        return response_payload["output_text"].strip()

    raise RuntimeError(
        "Could not find assistant message text in model response output."
    )


def parse_json_from_text(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                return value
            except json.JSONDecodeError:
                continue
        raise


def process_pdf(
    path: Path,
    config: dict[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "success",
        "data": None,
    }

    try:
        image_parts = render_pdf_as_image_parts(path)
        response_text = request_extraction(image_parts, config)
        extracted = parse_json_from_text(response_text)
        if not isinstance(extracted, dict):
            raise RuntimeError("Model returned JSON, but the top-level value is not an object.")
        result["data"] = extracted
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        return result


def write_output(output_path: Path, results: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    started_at = time.perf_counter()
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_config()
    concurrency = get_int_env("MODEL_CONCURRENCY", DEFAULT_CONCURRENCY)
    dpi = DEFAULT_PDF_RENDER_DPI
    paper_dir = resolve_project_path("01_paper_preprocess/paper")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = resolve_project_path(f"01_paper_preprocess/paper_{timestamp}.json")
    timeout = DEFAULT_TIMEOUT_SECONDS
    retries = DEFAULT_MAX_RETRIES

    print(f"Paper directory: {format_path(paper_dir)}")
    print(f"Output file: {format_path(output_path)}")
    print(f"Model: {config['model']}")
    print(
        f"Concurrency: {concurrency}; timeout: {timeout}s; "
        f"retries: {retries}; render DPI: {dpi}; "
    )

    pdf_files = find_pdf_files(paper_dir)

    if not pdf_files:
        write_output(output_path, [])
        elapsed = time.perf_counter() - started_at
        print(f"No PDF files found in {format_path(paper_dir)}.")
        print(f"Wrote empty result to {format_path(output_path)}.")
        print(f"Finished in {elapsed:.1f}s.")
        return 0

    print(f"Found {len(pdf_files)} PDF file(s).")
    print("Converting PDF pages in memory and parsing PDFs...")
    results_by_path: dict[Path, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                process_pdf,
                path,
                config
            ): path
            for path in pdf_files
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing PDFs",
            unit="paper",
        ):
            path = futures[future]
            result = future.result()
            status = result.get("status", "unknown")
            relative_path = format_path(path)
            if status == "success":
                tqdm.write(f"[success] {relative_path}")
                data = result.get("data")
                if isinstance(data, dict):
                    results_by_path[path] = data
            else:
                tqdm.write(f"[failed] {relative_path}: {result.get('error', 'unknown error')}")

    results = [results_by_path[path] for path in pdf_files if path in results_by_path]
    write_output(output_path, results)
    failed_count = len(pdf_files) - len(results)
    elapsed = time.perf_counter() - started_at
    print("Extraction finished.")
    print(f"Parsed successfully: {len(results)}")
    print(f"Failed: {failed_count}")
    print(f"Output file: {format_path(output_path)}")
    print(f"Elapsed time: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

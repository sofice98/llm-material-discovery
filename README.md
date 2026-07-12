# llm-material-discovery

用于从 PDF 论文提取结构化材料数据，并生成、翻译和分析训练数据。

## 项目结构

```text
llm-material-discovery/
├─ 01_paper_preprocess/
│  ├─ paper/                  # PDF 输入目录
│  ├─ output/                 # 论文解析结果
│  ├─ extraction_prompt.md    # 论文提取提示词
│  └─ paper_preprocess.py     # PDF 预处理脚本
├─ 02_get_train_data/
│  ├─ prompts/                # CPT、SFT、DPO 提示词
│  ├─ output/                 # 训练数据及工具默认输出目录
│  ├─ get_train_data.py       # 训练数据生成脚本
│  └─ jsonl_tools.py          # JSONL 工具
├─ llm_client/                 # 统一的 Chat Completions 请求客户端
├─ .env.example
└─ requirements.txt
```

## 环境配置

```powershell
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并配置：

| 变量 | 说明 |
| --- | --- |
| `API_KEY` | 大模型 API 密钥 |
| `CHAT_COMPLETIONS_URL` | Chat Completions API 地址，默认 MiniMax `/v1/chat/completions` |
| `MODEL` | 模型名称 |
| `MODEL_CONCURRENCY` | 并发数，默认 `50` |
| `MODEL_TIMEOUT_SECONDS` | 模型请求超时秒数，默认 `600` |
| `MODEL_MAX_RETRIES` | 模型请求失败重试次数，默认 `2` |
| `TRANSLATE_MAX_OUTPUT_TOKENS` | 翻译单次最大输出 token，默认 `32768` |

旧 `.env` 中的 `RESPONSES_URL` 若以 `/responses` 结尾，会自动转换为对应的 `/chat/completions` 地址；建议后续改用 `CHAT_COMPLETIONS_URL`。

## 01_paper_preprocess

### `paper_preprocess.py`

将 PDF 各页渲染为图片，调用多模态大模型提取结构化数据。默认读取 `01_paper_preprocess/paper`，输出到 `01_paper_preprocess/output`。

```powershell
python 01_paper_preprocess/paper_preprocess.py
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--paper-dir PATH` | 否 | `01_paper_preprocess/paper` | PDF 输入目录，递归查找 `.pdf` |
| `--output-dir PATH` | 否 | `01_paper_preprocess/output` | JSON 输出目录 |

## 02_get_train_data

### `get_train_data.py`

根据结构化论文 JSON 和提示词生成 CPT、SFT 或 DPO 格式的 JSONL 训练数据。

```powershell
python 02_get_train_data/get_train_data.py --method SFT --input 01_paper_preprocess/output/paper_xxx.json
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--method {CPT,SFT,DPO}` | 是 | - | 训练数据格式和提示词目录 |
| `--input PATH` | 否 | 最新的 `paper_*.json` | 输入论文 JSON |
| `--output-dir PATH` | 否 | `02_get_train_data/output` | JSONL 和错误报告输出目录 |

### `jsonl_tools.py translate`

分批调用大模型，将 JSONL 的指定文本字段翻译为中文或英文。输出默认保存到 `02_get_train_data/output`。

```powershell
python 02_get_train_data/jsonl_tools.py translate input.jsonl --language en
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | 是 | - | 输入 JSONL |
| `--language {zh,en}` | 是 | - | 目标语言 |
| `-o, --output PATH` | 否 | `<input>_<language>_<timestamp>.jsonl` | 输出文件；相对路径以项目根目录为基准 |
| `--fields FIELD ...` | 否 | 常见训练文本字段 | 需要翻译的顶层字符串字段 |
| `--batch-size N` | 否 | `10` | 单次请求最多包含的样本数 |
| `--concurrency N` | 否 | `MODEL_CONCURRENCY` | API 并发数 |
| `--max-batch-chars N` | 否 | `30000` | 单批最大源字符数 |
| `--overwrite` | 否 | 关闭 | 允许覆盖已有输出文件 |

### `jsonl_tools.py merge`

按输入顺序合并多个 JSONL。输出默认保存到 `02_get_train_data/output`。

```powershell
python 02_get_train_data/jsonl_tools.py merge a.jsonl b.jsonl
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `inputs` | 是 | - | 一个或多个输入 JSONL |
| `-o, --output PATH` | 否 | `merged_<timestamp>.jsonl` | 输出文件；相对路径以项目根目录为基准 |
| `--deduplicate` | 否 | 关闭 | 删除内容相同的 JSON 记录 |
| `--overwrite` | 否 | 关闭 | 允许覆盖已有输出文件 |

### `jsonl_tools.py count`

统计指定 JSONL 中有效 JSON 对象的样本数。

```powershell
python 02_get_train_data/jsonl_tools.py count train.jsonl
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | 是 | - | 输入 JSONL |

### `jsonl_tools.py task-stats`

统计 SFT JSONL 中各 `task_type` 的样本数量和比例。

```powershell
python 02_get_train_data/jsonl_tools.py task-stats train.jsonl
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | 是 | - | 输入 SFT JSONL |

### `jsonl_tools.py tokens`

使用 `tiktoken` 统计 JSONL token 总数及每条样本的平均值、最小值和最大值。

```powershell
python 02_get_train_data/jsonl_tools.py tokens train.jsonl
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | 是 | - | 输入 JSONL |
| `--model NAME` | 否 | - | 根据模型名称选择 tiktoken 编码 |
| `--encoding NAME` | 否 | `cl100k_base` | 指定 tiktoken 编码，与 `--model` 互斥 |
| `--fields FIELD ...` | 否 | 完整 JSON 记录 | 只统计指定顶层字段 |

所有相对输入路径均以项目根目录为基准。可通过以下命令查看实时帮助：

```powershell
python <脚本路径> --help
```

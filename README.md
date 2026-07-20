# llm-material-discovery

用于从 PDF 论文提取结构化材料数据，并生成、翻译和分析训练数据。

# 项目结构

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
├─ 03_train_model/            # Qwen3.5-9B CPT + SFT QLoRA 训练与合并脚本
├─ llm_client/                 # 统一的 Responses API 请求客户端
├─ .env.example
└─ requirements.txt
```

# 环境配置

```powershell
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并配置：

| 变量 | 说明 |
| --- | --- |
| `API_KEY` | 大模型 API 密钥 |
| `MODEL_URL` | Responses API 地址，例如 `https://api.openai.com/v1/responses`；该值会原样请求，请配置 `/responses` 端点 |
| `MODEL` | 模型名称 |
| `MODEL_TIMEOUT_SECONDS` | 模型请求超时秒数，默认 |
| `MODEL_MAX_RETRIES` | 模型请求失败重试次数，默认 |
| `QINIU_ACCESS_KEY` | 七牛云 AccessKey |
| `QINIU_SECRET_KEY` | 七牛云 SecretKey |
| `QINIU_BUCKET` | 七牛云对象存储空间名称 |
| `QINIU_PUBLIC_DOMAIN` | 七牛云空间的公网域名（含协议），例如 `http://123.hd-bkt.clouddn.com` |
| `QINIU_KEY_PREFIX` | 上传图片的对象键前缀，默认 `pdf-pages` |

# 使用说明

先从论文提取预处理JSON格式数据，再生成训练数据，最后可对训练数据进行翻译、合并、统计等操作。

## 01_paper_preprocess

### `paper_preprocess.py`

将 PDF 各页渲染为图片，上传到七牛云公网空间，再以图片 URL 调用多模态大模型提取结构化数据。默认读取 `01_paper_preprocess/paper`，输出到 `01_paper_preprocess/output`。

```powershell
python 01_paper_preprocess/paper_preprocess.py
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--paper-dir PATH` | 否 | `01_paper_preprocess/paper` | PDF 输入目录，递归查找 `.pdf` |
| `--output-dir PATH` | 否 | `01_paper_preprocess/output` | JSON 输出目录 |
| `--pages-per-request N` | 否 | `20` | 每次大模型请求包含的最大 PDF 页数 |
| `--image-max-mb N` | 否 | `10` | 单张渲染 PDF 页面图片的最大大小（MB）；超出时不上传也不请求模型 |
| `--concurrency N` | 否 | `10` | 大模型请求并发数 |

## 02_get_train_data

### `get_train_data.py`

根据预处理 JSON 和提示词生成 CPT、SFT 或 DPO 格式的 JSONL 训练数据。

```powershell
python 02_get_train_data/get_train_data.py --method SFT --input 01_paper_preprocess/output/paper_xxx.json
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--method {CPT,SFT,DPO}` | 是 | - | 训练数据格式和提示词目录 |
| `--input PATH` | 是 | - | 输入预处理JSON路径 |
| `--concurrency N` | 否 | `10` | 大模型请求并发数 |

JSONL 输出固定保存到 `02_get_train_data/output`；错误信息直接输出到控制台。

### `jsonl_tools.py translate`

分批调用大模型，将 JSONL 的指定文本字段翻译为中文或英文。输出默认保存到 `02_get_train_data/output`。

```powershell
python 02_get_train_data/jsonl_tools.py translate input.jsonl --language en
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | 是 | - | 输入 JSONL |
| `--language {zh,en}` | 是 | - | 目标语言 |
| `--batch-size N` | 否 | `10` | 单次请求最多包含的样本数 |
| `--concurrency N` | 否 | `10` | API 并发数 |
| `--max-batch-chars N` | 否 | `30000` | 单批最大源字符数 |

翻译固定处理 `instruction`、`input`、`output`、`text`、`prompt`、`chosen` 和 `rejected` 顶层字符串字段，并始终生成一个新的带时间戳输出文件。

### `jsonl_tools.py merge`

按输入顺序合并多个 JSONL。输出默认保存到 `02_get_train_data/output`。

```powershell
python 02_get_train_data/jsonl_tools.py merge a.jsonl b.jsonl
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `inputs` | 是 | - | 一个或多个输入 JSONL |
| `-o, --output PATH` | 否 | `merged_<timestamp>.jsonl` | 输出文件；相对路径以项目根目录为基准 |

### `jsonl_tools.py count`

统计指定 JSONL或JSON 中有效 JSON 对象的样本数；也支持顶层为数组的 `.json` 文件。

```powershell
python 02_get_train_data/jsonl_tools.py count train.jsonl
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | 是 | - | 输入 JSONL，或顶层为数组的 JSON 文件 |

传入 `.json` 文件时，顶层必须是数组；否则命令会报错并提示改用 JSONL 或提供数组格式的 JSON 文件。

### `jsonl_tools.py task-stats`

统计 SFT JSONL 中各 `task_type` 的样本数量和比例。

```powershell
python 02_get_train_data/jsonl_tools.py task-stats train.jsonl
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | 是 | - | 输入 SFT JSONL |

### `jsonl_tools.py tokens`

使用 Hugging Face `transformers` 加载 `Qwen/Qwen3.6-27B` tokenizer，统计 JSONL 文件每一行原始内容的 token 总数及每行的平均值、最小值和最大值。

```powershell
python 02_get_train_data/jsonl_tools.py tokens train.jsonl
```

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | 是 | - | 输入 JSONL |
| `--tokenizer-model NAME` | 否 | `Qwen/Qwen3.6-27B` | Hugging Face tokenizer 模型名称 |

所有相对输入路径均以项目根目录为基准。可通过以下命令查看实时帮助：

```powershell
python <脚本路径> --help
```

## 03_train_model

### Qwen3.5-9B 单卡 CPT + SFT QLoRA 训练

本目录保留 `dataset_info.json` 和量化程序 `quantize_final_model_4bit.py`。以下命令直接调用 Hugging Face、LLaMA-Factory、Python 和 vLLM，不需要 Shell 脚本。请将 `cpt.jsonl` 和 `sft.jsonl` 放入此目录，并在此目录中执行全部命令。所有模型、adapter 和量化结果均保存到 `output/`。

### 服务器环境

```bash
python3 -m venv output/.venv_train
source output/.venv_train/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -U llamafactory huggingface_hub bitsandbytes
```

若模型访问需要认证，执行 `hf auth login`。以下所有训练命令使用 GPU `0`、4bit NF4 QLoRA、`CUTOFF_LEN=2048`、每卡 batch size `1` 和梯度累积 `16`。

#### 阶段 1：CPT QLoRA 训练

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True hf download Qwen/Qwen3.5-9B --local-dir output/Qwen3.5-9B
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True llamafactory-cli train --stage pt --do_train true --model_name_or_path output/Qwen3.5-9B --dataset cpt --dataset_dir . --template empty --finetuning_type lora --lora_target all --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --quantization_bit 4 --quantization_method bnb --quantization_type nf4 --double_quantization true --cutoff_len 2048 --packing true --learning_rate 5e-5 --num_train_epochs 1.0 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --gradient_checkpointing true --dataloader_num_workers 4 --preprocessing_num_workers 4 --bf16 true --flash_attn auto --lr_scheduler_type cosine --warmup_ratio 0.03 --logging_steps 5 --save_strategy steps --save_steps 500 --save_total_limit 2 --plot_loss true --output_dir output/cpt_lora --report_to none
```

#### 阶段 2：合并 CPT adapter

合并阶段 1 的 adapter 到完整 BF16 基础模型。

```bash
llamafactory-cli export --model_name_or_path output/Qwen3.5-9B --adapter_name_or_path output/cpt_lora --template qwen3 --finetuning_type lora --export_dir output/cpt_merged --export_size 4 --export_device cpu --export_legacy_format false
```

#### 阶段 3：SFT QLoRA 训练

使用阶段 2 的完整 CPT 模型作为基础模型。

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True llamafactory-cli train --stage sft --do_train true --model_name_or_path output/cpt_merged --dataset sft --dataset_dir . --template qwen3 --finetuning_type lora --lora_target all --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --quantization_bit 4 --quantization_method bnb --quantization_type nf4 --double_quantization true --cutoff_len 2048 --learning_rate 1e-5 --num_train_epochs 3.0 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --gradient_checkpointing true --dataloader_num_workers 4 --preprocessing_num_workers 4 --val_size 0.05 --eval_strategy steps --eval_steps 100 --bf16 true --flash_attn auto --lr_scheduler_type cosine --warmup_ratio 0.03 --logging_steps 5 --save_strategy steps --save_steps 500 --save_total_limit 2 --plot_loss true --output_dir output/sft_lora --report_to none
```

#### 阶段 4：合并 SFT adapter

将阶段 3 的 adapter 合并到阶段 2 的完整模型。

```bash
llamafactory-cli export --model_name_or_path output/cpt_merged --adapter_name_or_path output/sft_lora --template qwen3 --finetuning_type lora --export_dir output/qwen3_5_9b_cpt_sft_merged --export_size 4 --export_device cpu --export_legacy_format false
```

#### 阶段 5：GPTQ 4bit 量化

量化完整 BF16 最终模型，不量化 QLoRA adapter。

```bash
pip install -U llm-compressor datasets
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python quantize_final_model_4bit.py output/qwen3_5_9b_cpt_sft_merged sft.jsonl output/qwen3_5_9b_cpt_sft_gptq_w4a16 --calibration-samples 64 --calibration-max-seq-len 1024 --max-shard-size 3900MB
```

#### 阶段 6：vLLM 部署

部署阶段 5 的完整 GPTQ 模型，不能直接部署 QLoRA adapter。

```bash
python3 -m venv output/.venv_vllm
source output/.venv_vllm/bin/activate
pip install -U pip vllm
export VLLM_API_KEY="$(openssl rand -hex 32)"
vllm serve output/qwen3_5_9b_cpt_sft_gptq_w4a16 --host 0.0.0.0 --port 8000 --api-key "$VLLM_API_KEY" --served-model-name qwen3_5_9b_material --tensor-parallel-size 1 --dtype auto --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --disable-log-requests --quantization compressed-tensors
```

调用示例：

```bash
curl http://SERVER_IP:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_5_9b_material",
    "messages": [{"role": "user", "content": "请说明该材料的主要性能。"}],
    "temperature": 0.2
  }'
```

需在云厂商安全组和服务器防火墙中放通 TCP `8000` 才能被公网访问。API Key 只负责鉴权，不提供 TLS 加密；生产环境应使用 Nginx 或 Caddy 反向代理并配置 HTTPS，且不应将生成的 Key 写入脚本、Git 仓库或聊天记录。

### 磁盘容量

完整流程会同时保留基础模型、CPT 合并模型和最终模型，还需预留 LoRA adapter、检查点、JSONL、日志和导出余量。请根据所选模型的实际大小规划数据盘容量。

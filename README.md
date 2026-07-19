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
├─ 03_train_model/            # Qwen3.6-27B CPT + SFT LoRA 训练与合并脚本
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

### Qwen3.6-27B 单卡 CPT + SFT LoRA 训练

本目录有六个独立脚本：

1. `01_train_cpt.sh`：训练 CPT LoRA；数据文件路径作为第一个命令行参数传入。
2. `02_merge_cpt.sh`：合并指定的 CPT LoRA；LoRA 路径作为第一个命令行参数传入。
3. `03_train_sft.sh`：基于指定 CPT 合并模型训练 SFT LoRA。
4. `04_merge_sft.sh`：合并指定的 SFT LoRA，生成最终模型。
5. `03_quantize_final_model_4bit.sh`：将指定的最终 BF16 模型进行 GPTQ W4A16 量化。
6. `04_serve_vllm_public.sh`：以 vLLM 启动指定模型的 OpenAI 兼容公网 API。

默认参数针对单张 RTX PRO 6000 96GB、22 核 CPU、110GB 内存：BF16、`CUTOFF_LEN=6144`、每卡 batch size `1`、梯度累积 `8`、LoRA `r=64`。

### 上传到服务器

服务器上只需一个工作目录，建议使用数据盘路径，例如 `/data/qwen3_6_27b_train`。将以下六个脚本和训练数据放入同一目录：

```text
/data/qwen3_6_27b_train/
├── 01_train_cpt.sh
├── 02_merge_cpt.sh
├── 03_train_sft.sh
├── 04_merge_sft.sh
├── 03_quantize_final_model_4bit.sh
├── 04_serve_vllm_public.sh
├── cpt.jsonl
└── sft.jsonl
```

在服务器创建目录：

```bash
mkdir -p /data/qwen3_6_27b_train
```

从本地电脑上传。将 `USER`、`SERVER_IP` 和本地 JSONL 实际文件名替换为自己的值：

```bash
scp 03_train_model/01_train_cpt.sh USER@SERVER_IP:/data/qwen3_6_27b_train/
scp 03_train_model/02_merge_cpt.sh USER@SERVER_IP:/data/qwen3_6_27b_train/
scp 03_train_model/03_train_sft.sh USER@SERVER_IP:/data/qwen3_6_27b_train/
scp 03_train_model/04_merge_sft.sh USER@SERVER_IP:/data/qwen3_6_27b_train/
scp 03_train_model/03_quantize_final_model_4bit.sh USER@SERVER_IP:/data/qwen3_6_27b_train/
scp 03_train_model/04_serve_vllm_public.sh USER@SERVER_IP:/data/qwen3_6_27b_train/
scp 02_get_train_data/output/train_cpt_xxx.jsonl USER@SERVER_IP:/data/qwen3_6_27b_train/cpt.jsonl
scp 02_get_train_data/output/train_sft_xxx.jsonl USER@SERVER_IP:/data/qwen3_6_27b_train/sft.jsonl
```

若没有挂载数据盘，可将 `/data/qwen3_6_27b_train` 改为 `~/qwen3_6_27b_train`；该路径需要至少 300GB 可用空间，推荐 500GB。

### 服务器环境

登录服务器后，在工作目录创建 Python 环境。RTX PRO 6000 建议使用支持 CUDA 12.8 的 PyTorch；以下为示例，实际安装命令应与服务器 CUDA 驱动和 Python 版本匹配。

```bash
cd /data/qwen3_6_27b_train
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -U llamafactory huggingface_hub
```

验证 GPU 和 BF16 可用：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"
```

若模型访问需要认证，登录 Hugging Face：

```bash
hf auth login
```

### 单卡直接训练

赋予脚本执行权限后，按顺序在后台启动。训练脚本直接读取传入的 JSONL，不会移动或复制数据文件。每次训练或合并生成的输出目录均带有时间戳；确认前一阶段的日志成功结束后，再将其输出路径传给下一阶段。

训练与合并参数可通过命令行路径或同名环境变量覆盖。未说明的项使用脚本默认值。

| 阶段 | 参数 | 默认值 | 说明 |
| --- | --- | --- | --- |
| CPT 训练 | 第一个位置参数 / `CPT_DATA` | 必填 | CPT JSONL 文件路径 |
| CPT 训练 | `MODEL_ID` | `Qwen/Qwen3.6-27B` | Hugging Face 基础模型名称 |
| CPT 训练 | `BASE_MODEL_DIR` | `${RUN_ROOT}/models/Qwen3.6-27B` | 基础模型本地目录，不存在时自动下载 |
| CPT 训练 | `CPT_LORA_DIR` | `${RUN_DIR}/cpt_lora_${RUN_ID}` | CPT LoRA 输出目录 |
| CPT 训练 | `CUTOFF_LEN` | `6144` | 单条样本最大 token 数 |
| CPT 训练 | `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | 每张 GPU 的训练 batch size |
| CPT/SFT 训练 | `GRADIENT_ACCUMULATION_STEPS` | `8` | 梯度累积步数 |
| CPT/SFT 训练 | `DATALOADER_NUM_WORKERS` | `8` | DataLoader 工作进程数 |
| CPT/SFT 训练 | `PREPROCESSING_NUM_WORKERS` | `8` | 数据预处理工作进程数 |
| CPT/SFT 训练 | `LORA_RANK` / `LORA_ALPHA` / `LORA_DROPOUT` | `64` / `128` / `0.05` | LoRA 配置 |
| CPT 训练 | `CPT_EPOCHS` / `CPT_LEARNING_RATE` | `1.0` / `5e-5` | CPT 训练轮数和学习率 |
| SFT 训练 | 第一个位置参数 / `SFT_DATA` | 必填 | SFT JSONL 文件路径 |
| SFT 训练 | 第二个位置参数 / `CPT_MERGED_DIR` | 必填 | CPT 合并模型目录 |
| SFT 训练 | `SFT_LORA_DIR` | `${RUN_DIR}/sft_lora_${RUN_ID}` | SFT LoRA 输出目录 |
| SFT 训练 | `CUTOFF_LEN` / `PER_DEVICE_TRAIN_BATCH_SIZE` | `6144` / `1` | 最大 token 数和每卡训练 batch size |
| SFT 训练 | `SFT_EPOCHS` / `SFT_LEARNING_RATE` | `3.0` / `1e-5` | SFT 训练轮数和学习率 |
| SFT 训练 | `SFT_VAL_SIZE` / `EVAL_STEPS` | `0.05` / `100` | 验证集比例和评估间隔 |
| CPT 合并 | 第一个位置参数 / `CPT_LORA_DIR` | 必填 | 待合并的 CPT LoRA 目录 |
| CPT 合并 | `BASE_MODEL_DIR` | `${RUN_ROOT}/models/Qwen3.6-27B` | 合并时使用的基础模型目录 |
| CPT 合并 | `CPT_MERGED_DIR` | `${RUN_DIR}/cpt_merged_${RUN_ID}` | CPT 合并模型输出目录 |
| SFT 合并 | 第一个位置参数 / `SFT_LORA_DIR` | 必填 | 待合并的 SFT LoRA 目录 |
| SFT 合并 | 第二个位置参数 / `CPT_MERGED_DIR` | 必填 | SFT 使用的 CPT 合并模型目录 |
| SFT 合并 | `FINAL_MODEL_DIR` | `${RUN_DIR}/qwen3_6_27b_cpt_sft_merged_${RUN_ID}` | 最终模型输出目录 |
| 两次合并 | `EXPORT_DEVICE` / `EXPORT_SIZE` | `cpu` / `4` | 导出设备和单个权重分片最大大小（GB） |
| 所有训练和合并 | `RUN_ROOT` / `RUN_DIR` / `RUN_ID` | 脚本目录 / `${RUN_ROOT}/runs` / 当前时间 | 工作根目录、结果根目录和输出目录时间戳 |
| 所有训练 | `CUDA_VISIBLE_DEVICES` / `PYTORCH_CUDA_ALLOC_CONF` | `0` / `expandable_segments:True` | 使用的 GPU 和 PyTorch 显存分配配置 |

```bash
cd /data/qwen3_6_27b_train
chmod +x 01_train_cpt.sh 02_merge_cpt.sh 03_train_sft.sh 04_merge_sft.sh 05_quantize_final_model_4bit.sh 06_serve_vllm_public.sh

# 完成后从 cpt_train.log 取得 cpt_lora_YYYYMMDD_HHMMSS 路径，再执行下一条。
nohup env CUDA_VISIBLE_DEVICES=0 bash 01_train_cpt.sh /data/qwen3_6_27b_train/cpt.jsonl > cpt_train.log 2>&1 &
echo "CPT training PID: $!"

# 完成后从 cpt_merge.log 取得 cpt_merged_YYYYMMDD_HHMMSS 路径，再执行下一条。
nohup env CUDA_VISIBLE_DEVICES=0 bash 02_merge_cpt.sh /data/qwen3_6_27b_train/runs/cpt_lora_YYYYMMDD_HHMMSS > cpt_merge.log 2>&1 &
echo "CPT merge PID: $!"

# 使用上一阶段的 CPT 合并模型路径。
nohup env CUDA_VISIBLE_DEVICES=0 bash 03_train_sft.sh /data/qwen3_6_27b_train/sft.jsonl /data/qwen3_6_27b_train/runs/cpt_merged_YYYYMMDD_HHMMSS > sft_train.log 2>&1 &
echo "SFT training PID: $!"

# 使用上一阶段的 SFT LoRA 和 CPT 合并模型路径。
nohup env CUDA_VISIBLE_DEVICES=0 bash 04_merge_sft.sh /data/qwen3_6_27b_train/runs/sft_lora_YYYYMMDD_HHMMSS /data/qwen3_6_27b_train/runs/cpt_merged_YYYYMMDD_HHMMSS > sft_merge.log 2>&1 &
echo "SFT merge PID: $!"

# 查看任一后台任务日志。
tail -f cpt_train.log
```

默认生成的文件位置：

```text
/data/qwen3_6_27b_train/
├── models/Qwen3.6-27B/                    # 下载的 55.6GB 基础模型
└── runs/
    ├── cpt_lora_YYYYMMDD_HHMMSS/           # CPT LoRA adapter、检查点、loss 图
    ├── cpt_merged_YYYYMMDD_HHMMSS/         # CPT 合并模型，SFT 的输入
    ├── sft_lora_YYYYMMDD_HHMMSS/           # SFT LoRA adapter、检查点、loss 图和评估结果
    └── qwen3_6_27b_cpt_sft_merged_YYYYMMDD_HHMMSS/ # 最终合并模型
```

最终模型路径为 `/data/qwen3_6_27b_train/runs/qwen3_6_27b_cpt_sft_merged`，目录中包含权重和 tokenizer，可直接用 `transformers` 加载。

默认 `CUTOFF_LEN=6144`。若 `nvidia-smi` 显示仍有显存余量，可用 `nohup env CUDA_VISIBLE_DEVICES=0 CUTOFF_LEN=8192 bash 01_train_cpt.sh /path/to/cpt.jsonl > cpt_train.log 2>&1 &` 和相同方式运行 SFT 进行测试；若出现 CUDA OOM，依次回退为 `4096`、`3072`。不建议未经最长样本测试就将每卡 batch size 增加至 2。

合并默认在 CPU 上进行。110GB 内存合并 55.6GB BF16 模型的余量较紧，合并期间不要同时运行其他大型任务。若 CPU 合并内存不足，可在命令前加上 `EXPORT_DEVICE=auto`，使用 GPU 进行合并。

两个合并脚本默认设置 `EXPORT_SIZE=4`，所以合并模型的 safetensors 权重按最大约 4GB 分片。若需要调整可在运行前设置，例如 `nohup env EXPORT_SIZE=3 bash 02_merge_cpt.sh /path/to/cpt_lora > cpt_merge.log 2>&1 &`。注意单个 tensor 极大时无法再切分；Qwen3.6-27B 的权重 tensor 小于该限制，因此会正常分片。

### 4bit GPTQ 量化

量化脚本使用 SFT JSONL 中的 128 条代表性样本作为校准数据，生成 vLLM 可识别的 GPTQ W4A16 `compressed-tensors` 模型。该过程需要 GPU，运行前在训练环境安装依赖：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| 第一个位置参数 / `FINAL_MODEL_DIR` | 必填 | 待量化的最终模型目录 |
| 第二个位置参数 / `CALIBRATION_DATA` | 必填 | 校准用 SFT JSONL 文件路径 |
| `QUANTIZED_MODEL_DIR` | `${RUN_ROOT}/runs/qwen3_6_27b_cpt_sft_gptq_w4a16_${RUN_ID}` | 量化模型输出目录 |
| `CALIBRATION_SAMPLES` | `128` | 校准样本数 |
| `CALIBRATION_MAX_SEQ_LEN` | `2048` | 校准的最大序列长度 |
| `MAX_SHARD_SIZE` | `3900MB` | 导出 safetensors 的最大分片大小 |
| `RUN_ROOT` / `RUN_ID` | 脚本目录 / 当前时间 | 工作根目录和输出目录时间戳 |
| `CUDA_VISIBLE_DEVICES` / `PYTORCH_CUDA_ALLOC_CONF` | `0` / `expandable_segments:True` | 使用的 GPU 和 PyTorch 显存分配配置 |

```bash
cd /data/qwen3_6_27b_train
source .venv/bin/activate
pip install -U llm-compressor datasets
chmod +x 05_quantize_final_model_4bit.sh
nohup env CUDA_VISIBLE_DEVICES=0 bash 05_quantize_final_model_4bit.sh /data/qwen3_6_27b_train/runs/qwen3_6_27b_cpt_sft_merged_YYYYMMDD_HHMMSS /data/qwen3_6_27b_train/sft.jsonl > quantize_4bit.log 2>&1 &
echo "Quantization PID: $!"
```

默认输出目录为：

```text
/data/qwen3_6_27b_train/runs/qwen3_6_27b_cpt_sft_gptq_w4a16_YYYYMMDD_HHMMSS/
```

量化权重以 `MAX_SHARD_SIZE=3900MB` 保存，低于 4GB。若 SFT 数据分布不够代表真实请求，请通过 `CALIBRATION_DATA=/path/to/representative_sft.jsonl` 传入更合适的校准 JSONL；可将 `CALIBRATION_SAMPLES` 提高到 256，但量化时间会更长。

### vLLM 公网服务

建议使用独立环境安装 vLLM，避免与训练环境的依赖版本冲突：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| 第一个位置参数 / `MODEL_PATH` | 必填 | 要部署的模型目录 |
| `VLLM_API_KEY` | 必填 | OpenAI 兼容接口的鉴权密钥 |
| `SERVED_MODEL_NAME` | `qwen3_6_27b_material` | API 请求中使用的模型名称 |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | 服务监听地址和端口 |
| `MAX_MODEL_LEN` | `8192` | 最大上下文长度 |
| `GPU_MEMORY_UTILIZATION` | `0.90` | vLLM 可使用的 GPU 显存比例 |
| `VLLM_QUANTIZATION` | 空 | 可选量化后端；自动识别失败时设为 `compressed-tensors` |

```bash
cd /data/qwen3_6_27b_train
python3 -m venv .venv_vllm
source .venv_vllm/bin/activate
pip install -U pip vllm
chmod +x 06_serve_vllm_public.sh
export VLLM_API_KEY="$(openssl rand -hex 32)"
nohup bash 06_serve_vllm_public.sh /data/qwen3_6_27b_train/runs/qwen3_6_27b_cpt_sft_gptq_w4a16_YYYYMMDD_HHMMSS > vllm.log 2>&1 &
```

默认服务监听 `0.0.0.0:8000`，模型名为 `qwen3_6_27b_material`，提供 OpenAI 兼容的 `/v1/chat/completions` 接口。vLLM 会从量化模型配置自动识别 GPTQ；若当前 vLLM 版本未自动识别，可在启动前设置 `VLLM_QUANTIZATION=compressed-tensors`。

调用示例：

```bash
curl http://SERVER_IP:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3_6_27b_material",
    "messages": [{"role": "user", "content": "请说明该材料的主要性能。"}],
    "temperature": 0.2
  }'
```

需在云厂商安全组和服务器防火墙中放通 TCP `8000` 才能被公网访问。API Key 只负责鉴权，不提供 TLS 加密；生产环境应使用 Nginx 或 Caddy 反向代理并配置 HTTPS，且不应将生成的 Key 写入脚本、Git 仓库或聊天记录。

### 如何判断训练效果

训练损失下降只说明模型拟合了训练数据，并不等于实际效果提升。应同时看以下三项：

1. 实时监控：另开一个 SSH 窗口运行 `tail -f cpt_train.log`、`tail -f sft_train.log` 和 `nvidia-smi -l 2`。带时间戳的 `runs/cpt_lora_*` 与 `runs/sft_lora_*` 中会生成 `training_loss.png`，可下载后查看 loss 是否平稳下降。
2. SFT 验证损失：SFT 脚本会从传入的 SFT JSONL 随机留出 `5%` 作为验证集，并每 `100` step 记录 `eval_loss`。训练后查看 `sft_train.log` 中的 `eval_loss`，或对应 `runs/sft_lora_*/trainer_state.json`。若训练 loss 持续下降而 eval loss 连续上升，应减少 `SFT_EPOCHS` 或降低学习率。
3. 留出任务评测：训练前从 SFT 数据中保留 5% 到 10% 真实样本，不放入 `sft.jsonl`。对同一批问题分别使用基础模型、`runs/cpt_merged` 和最终模型生成答案，再按照你的材料任务标准评分，例如数值和单位是否正确、是否引用了给定条件、推荐方案是否可执行、是否出现编造内容。最终模型应在留出集上优于基础模型，而不是只在训练样本上复述得更好。

模型目录可通过 LLaMA-Factory 交互测试：

```bash
llamafactory-cli chat \
  --model_name_or_path /data/qwen3_6_27b_train/runs/qwen3_6_27b_cpt_sft_merged_YYYYMMDD_HHMMSS \
  --template qwen3
```

### 磁盘容量

Hugging Face 标示的基础模型为 55.6GB。完整流程会同时保留基础模型、CPT 合并模型和最终模型，权重约占 `55.6 x 3 = 166.8GB`，还需预留 LoRA adapter、检查点、JSONL、日志和导出余量。因此数据盘至少需要 300GB 可用空间，建议使用 500GB 或更大。

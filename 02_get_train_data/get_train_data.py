# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import ChatCompletionClient, LLMClientError, create_client


SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "01_paper_preprocess"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_CONCURRENCY = 50
DEFAULT_MAX_SAMPLE_CHARS = 8000
DEFAULT_MAX_OUTPUT_TOKENS = 65536

TASKS = {
    "01_forward_property_prediction": "forward_property_prediction",
    "02_inverse_composition_recommendation": "inverse_composition_recommendation",
    "03_inverse_process_optimization": "inverse_process_optimization",
    "04_inverse_scenario_adaptation": "inverse_scenario_adaptation",
    "05_application_scenario_classification": "application_scenario_classification",
    "06_synthesis_feasibility_prediction": "synthesis_feasibility_prediction",
    "07_anomaly_detection": "anomaly_detection",
    "08_literature_qa": "literature_qa",
    "09_mechanism_explanation": "mechanism_explanation",
}
REQUIRED_FIELDS = {
    "CPT": ("text",),
    "SFT": ("task_type", "instruction", "input", "output"),
    "DPO": ("prompt", "chosen", "rejected"),
}


class ConfigError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate merged LlamaFactory JSONL training data from paper JSON."
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str.upper,
        choices=("CPT", "SFT", "DPO"),
        help="Fine-tuning data format and prompt directory to use.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input paper JSON. Defaults to the newest 01_paper_preprocess/paper_*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for timestamped JSONL and error-report files.",
    )
    return parser.parse_args()


def get_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got: {raw}") from exc
    if value < 1:
        raise ConfigError(f"{name} must be at least 1, got: {value}")
    return value


def load_config() -> dict[str, Any]:
    config: dict[str, Any] = {
        "concurrency": get_positive_int_env("MODEL_CONCURRENCY", DEFAULT_CONCURRENCY),
        "max_sample_chars": get_positive_int_env(
            "TRAIN_MAX_SAMPLE_CHARS", DEFAULT_MAX_SAMPLE_CHARS
        ),
    }
    return config


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def find_latest_input() -> Path:
    candidates = sorted(
        DEFAULT_INPUT_DIR.glob("paper_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No paper_*.json files found in {display_path(DEFAULT_INPUT_DIR)}"
        )
    return candidates[0]


def load_papers(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Input JSON must contain an object or an array of objects.")
    papers = [item for item in payload if isinstance(item, dict)]
    if len(papers) != len(payload):
        raise ValueError("Every item in the input JSON array must be an object.")
    return papers


def load_prompts(method: str) -> dict[str, str]:
    method_dir = PROMPTS_DIR / method
    prompts: dict[str, str] = {}
    missing: list[str] = []
    for prompt_name in TASKS:
        path = method_dir / f"{prompt_name}.md"
        if not path.is_file():
            missing.append(str(path))
            continue
        prompt = path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ConfigError(f"Prompt file is empty: {path}")
        prompts[prompt_name] = prompt
    if missing:
        raise ConfigError("Missing prompt files: " + ", ".join(missing))
    return prompts


def parse_json_from_text(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:] if lines else lines
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
                pass
        raise


def request_samples(
    paper: dict[str, Any],
    prompt: str,
    method: str,
    task_type: str,
    config: dict[str, Any],
    client: ChatCompletionClient,
) -> list[dict[str, Any]]:
    paper_json = json.dumps(paper, ensure_ascii=False, separators=(",", ":"))
    format_hint = {
        "CPT": '[{"text":"..."}]',
        "SFT": (
            '[{"task_type":"' + task_type + '","instruction":"...",'
            '"input":"...","output":"..."}]'
        ),
        "DPO": '[{"prompt":"...","chosen":"...","rejected":"..."}]',
    }[method]
    user_text = (
        f"{prompt}\n\n"
        "硬性要求：\n"
        "1. 只能使用下方文献 JSON 中明确给出的事实，不得杜撰数值、条件或结论。\n"
        "2. 在证据充分且样本不重复的前提下尽可能生成多条数据。\n"
        f"3. 每条样本序列化后不得超过 {config['max_sample_chars']} 个字符。\n"
        "4. 只返回一个合法 JSON 数组，不要 Markdown 代码块或说明文字。\n"
        f"5. 数组元素格式必须为：{format_hint}\n\n"
        f"文献 JSON：\n{paper_json}"
    )
    response_text = client.complete(
        [{"role": "user", "content": user_text}],
        max_completion_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_type="adaptive",
    )
    parsed = parse_json_from_text(response_text)
    if isinstance(parsed, dict):
        for key in ("samples", "data", "items"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        raise RuntimeError("Model response JSON must be an array of samples.")
    return [item for item in parsed if isinstance(item, dict)]


def normalize_sample(
    sample: dict[str, Any], method: str, task_type: str, max_chars: int
) -> dict[str, str] | None:
    fields = REQUIRED_FIELDS[method]
    normalized: dict[str, str] = {}
    for field in fields:
        value = sample.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
        normalized[field] = value.strip()
    if method == "SFT":
        normalized["task_type"] = task_type
    if method == "DPO" and normalized["chosen"] == normalized["rejected"]:
        return None
    serialized = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > max_chars:
        return None
    return normalized


def process_job(
    paper_index: int,
    paper: dict[str, Any],
    prompt_name: str,
    prompt: str,
    method: str,
    config: dict[str, Any],
    client: ChatCompletionClient,
) -> dict[str, Any]:
    task_type = TASKS[prompt_name]
    try:
        raw_samples = request_samples(paper, prompt, method, task_type, config, client)
        samples = [
            normalized
            for sample in raw_samples
            if (
                normalized := normalize_sample(
                    sample, method, task_type, config["max_sample_chars"]
                )
            )
            is not None
        ]
        return {
            "paper_index": paper_index,
            "task": task_type,
            "samples": samples,
            "discarded": len(raw_samples) - len(samples),
        }
    except Exception as exc:
        return {"paper_index": paper_index, "task": task_type, "error": str(exc)}


def write_jsonl(path: Path, samples: list[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written_count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            written_count += 1
    return written_count


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    client = create_client()
    config = load_config()
    input_path = resolve_path(args.input) if args.input else find_latest_input()
    output_dir = resolve_path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    papers = load_papers(input_path)
    prompts = load_prompts(args.method)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"train_{args.method.lower()}_{timestamp}.jsonl"
    error_path = output_dir / f"train_{args.method.lower()}_{timestamp}_errors.json"

    print(f"Method: {args.method}")
    print(f"Input file: {display_path(input_path)}")
    print(f"Output file: {display_path(output_path)}")
    print(f"Model: {client.model}")
    print(
        f"Papers: {len(papers)}; tasks: {len(prompts)}; "
        f"API calls: {len(papers) * len(prompts)}; concurrency: {config['concurrency']}"
    )

    results: list[dict[str, Any]] = []
    jobs = [
        (paper_index, paper, prompt_name, prompt)
        for paper_index, paper in enumerate(papers)
        for prompt_name, prompt in prompts.items()
    ]
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
        futures = [
            executor.submit(
                process_job,
                paper_index,
                paper,
                prompt_name,
                prompt,
                args.method,
                config,
                client,
            )
            for paper_index, paper, prompt_name, prompt in jobs
        ]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating", unit="call"
        ):
            results.append(future.result())

    results.sort(key=lambda item: (item["paper_index"], item["task"]))
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    errors: list[dict[str, Any]] = []
    discarded = 0
    for result in results:
        if "error" in result:
            errors.append(result)
            continue
        discarded += result["discarded"]
        for sample in result["samples"]:
            key = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                merged.append(sample)

    written_count = write_jsonl(output_path, merged)
    if errors:
        error_path.write_text(
            json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    elapsed = time.perf_counter() - started_at
    print(f"Invalid/oversized samples discarded: {discarded}")
    print(f"Failed API calls: {len(errors)}")
    if errors:
        print(f"Error report: {display_path(error_path)}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Final output samples: {written_count}")
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigError, LLMClientError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        raise SystemExit(1)

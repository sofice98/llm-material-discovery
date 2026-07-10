# -*- coding: utf-8 -*-
from __future__ import annotations

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


EXTRACTION_PROMPT = """
你是一名材料科学文献数据抽取助手。请读取上传的 PDF 文献，抽取热喷涂/涂层/粉末相关结构化数据。

要求：
1. 只输出合法 JSON，不要输出 Markdown、代码块或解释文字。
2. 未在文献中明确出现的信息填 null，不要猜测。
3. 尽量保留原文数值、范围和单位；如果同一字段存在多个样品/涂层/实验组，用数组逐条给出，并保留样品或涂层编号。
4. 作者用字符串数组。化学成分按元素及质量分数 wt% 抽取。
5. DOI、年份、期刊、题名优先从首页、摘要页或元数据中识别。
6. JSON 的所有字段名必须使用下面给出的中文 key，不要输出英文 key。

请严格按以下 JSON 结构输出：
{
  "文献元数据": {
    "标题": null,
    "DOI": null,
    "发表年份": null,
    "期刊": null,
    "作者": []
  },
  "粉末属性": {
    "化学成分": [
      {
        "样品/粉末编号": null,
        "元素": null,
        "质量分数 wt%": null
      }
    ],
    "粉末制备方法": null,
    "粒径范围（μm）": null
  },
  "喷涂工艺参数": [
    {
      "样品/涂层编号": null,
      "喷涂技术类型": null,
      "大气等离子喷涂（APS）": {
        "电流（A）": null,
        "电压（V）": null,
        "功率（KW）": null,
        "主气流量（L/min）": null,
        "次气流量（L/min）": null,
        "喷涂距离（mm）": null,
        "送粉率（g/min）": null,
        "喷枪移动速度（mm/s）": null,
        "基体预热温度": null,
        "后处理工艺": null
      },
      "超音速火焰喷涂（HVOF）": {
        "氧气流量（L/min）": null,
        "燃料流量（L/min）": null,
        "燃烧比": null,
        "喷涂距离（mm）": null,
        "送粉率（g/min）": null,
        "喷枪移动速度（mm/s）": null,
        "基体预热温度": null,
        "后处理工艺": null
      },
      "电弧喷涂": {
        "电流": null,
        "电压": null,
        "气体压力": null,
        "喷涂距离（mm）": null,
        "送丝速度": null,
        "喷枪移动速度（mm/s）": null,
        "基体预热温度": null,
        "后处理工艺": null
      }
    }
  ],
  "微观组织结构": [
    {
      "样品/涂层编号": null,
      "孔隙率": null,
      "相组成": null,
      "涂层厚度": null,
      "界面结合状态": null
    }
  ],
  "力学性能参数": [
    {
      "样品/涂层编号": null,
      "硬度": null,
      "结合强度": null,
      "残余应力": null
    }
  ],
  "摩擦学性能": [
    {
      "样品/涂层编号": null,
      "摩擦系数": null,
      "磨损率": null,
      "磨损机理": null
    }
  ]
}
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
        "api_key": os.getenv("ARK_API_KEY", "").strip(),
        "files_url": os.getenv(
            "ARK_FILES_URL", "https://ark.cn-beijing.volces.com/api/v3/files"
        ).strip(),
        "responses_url": os.getenv(
            "ARK_RESPONSES_URL",
            "https://ark.cn-beijing.volces.com/api/v3/responses",
        ).strip(),
        "model": os.getenv("ARK_MODEL", "").strip(),
    }

    missing = [
        name
        for name, value in {
            "ARK_API_KEY": config["api_key"],
            "ARK_FILES_URL": config["files_url"],
            "ARK_RESPONSES_URL": config["responses_url"],
            "ARK_MODEL": config["model"],
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
    timeout: int,
    retries: int,
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, timeout=timeout, **kwargs)
            if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            if getattr(exc, "response", None) is not None and exc.response is not None:
                body = exc.response.text[:1000]
                raise RuntimeError(f"API request failed: {exc}; response body: {body}") from exc
            raise RuntimeError(f"API request failed: {exc}") from exc
    raise RuntimeError(f"API request failed: {last_error}")


def upload_pdf(path: Path, config: dict[str, str], timeout: int, retries: int) -> str:
    headers = {"Authorization": f"Bearer {config['api_key']}"}
    with path.open("rb") as file_obj:
        response = post_with_retries(
            config["files_url"],
            headers=headers,
            timeout=timeout,
            retries=retries,
            data={"purpose": "user_data"},
            files={"file": (path.name, file_obj, "application/pdf")},
        )
    payload = response.json()
    file_id = (
        payload.get("id")
        or payload.get("file_id")
        or payload.get("data", {}).get("id")
        or payload.get("data", {}).get("file_id")
    )
    if not file_id:
        raise RuntimeError(f"Files API did not return a file id: {payload}")
    return str(file_id)


def request_extraction(
    file_id: str,
    config: dict[str, str],
    timeout: int,
    retries: int,
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
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
    }
    response = post_with_retries(
        config["responses_url"],
        headers=headers,
        timeout=timeout,
        retries=retries,
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
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "success",
        "data": None,
    }

    try:
        file_id = upload_pdf(path, config, timeout, retries)
        time.sleep(5)
        response_text = request_extraction(file_id, config, timeout, retries)
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
    concurrency = get_int_env("PAPER_CONCURRENCY", DEFAULT_CONCURRENCY)
    paper_dir = resolve_project_path("01paper-preprocess/paper")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = resolve_project_path(f"01paper-preprocess/paper_{timestamp}.json")
    timeout = DEFAULT_TIMEOUT_SECONDS
    retries = DEFAULT_MAX_RETRIES

    print(f"Paper directory: {format_path(paper_dir)}")
    print(f"Output file: {format_path(output_path)}")
    print(f"Model: {config['model']}")
    print(f"Concurrency: {concurrency}; timeout: {timeout}s; retries: {retries}")

    pdf_files = find_pdf_files(paper_dir)

    if not pdf_files:
        write_output(output_path, [])
        elapsed = time.perf_counter() - started_at
        print(f"No PDF files found in {format_path(paper_dir)}.")
        print(f"Wrote empty result to {format_path(output_path)}.")
        print(f"Finished in {elapsed:.1f}s.")
        return 0

    print(f"Found {len(pdf_files)} PDF file(s).")
    print("Uploading and parsing PDFs...")
    results_by_path: dict[Path, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                process_pdf,
                path,
                config,
                timeout,
                retries,
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

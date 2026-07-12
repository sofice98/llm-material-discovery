# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import ChatCompletionClient, LLMClientError, create_client


DEFAULT_PAPER_DIR = "01_paper_preprocess/paper"
DEFAULT_OUTPUT_DIR = "01_paper_preprocess/output"
DEFAULT_CONCURRENCY = 50
DEFAULT_PDF_RENDER_DPI = 300
IMAGE_MIME_TYPE = "image/png"
IMAGE_FORMAT = "png"


EXTRACTION_PROMPT = Path(__file__).with_name("extraction_prompt.md").read_text(encoding="utf-8").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured coating data from PDF papers."
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path(DEFAULT_PAPER_DIR),
        help=(
            "Directory containing PDF papers. Relative paths are resolved from the "
            f"project root (default: {DEFAULT_PAPER_DIR})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory for timestamped output JSON files. Relative paths are "
            f"resolved from the project root (default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    return parser.parse_args()


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


def find_pdf_files(paper_dir: Path) -> list[Path]:
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")
    if not paper_dir.is_dir():
        raise NotADirectoryError(f"Paper path is not a directory: {paper_dir}")
    return sorted(path for path in paper_dir.rglob("*.pdf") if path.is_file())


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
                    "type": "image_url",
                    "image_url": {"url": f"data:{IMAGE_MIME_TYPE};base64,{encoded}"},
                }
            )

    return image_parts


def request_extraction(
    image_parts: list[dict[str, Any]],
    client: ChatCompletionClient,
) -> str:
    return client.complete(
        [
            {
                "role": "user",
                "content": [
                    *image_parts,
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            },
        ],
        max_completion_tokens=524288,
        thinking_type="adaptive",
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
    client: ChatCompletionClient,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "success",
        "data": None,
    }

    try:
        image_parts = render_pdf_as_image_parts(path)
        response_text = request_extraction(image_parts, client)
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
    args = parse_args()
    client = create_client()
    concurrency = get_int_env("MODEL_CONCURRENCY", DEFAULT_CONCURRENCY)
    dpi = DEFAULT_PDF_RENDER_DPI
    paper_dir = resolve_project_path(args.paper_dir)
    output_dir = resolve_project_path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"paper_{timestamp}.json"

    print(f"Paper directory: {format_path(paper_dir)}")
    print(f"Output file: {format_path(output_path)}")
    print(f"Model: {client.model}")
    print(
        f"Concurrency: {concurrency}; render DPI: {dpi}; "
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
                client
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
    except LLMClientError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

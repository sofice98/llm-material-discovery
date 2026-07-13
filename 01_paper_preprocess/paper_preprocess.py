# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import (
    LLMClientError,
    QiniuUploader,
    ResponsesClient,
    create_client,
    create_qiniu_uploader,
    object_key_for_pdf_page,
)


DEFAULT_PAPER_DIR = "01_paper_preprocess/paper"
DEFAULT_OUTPUT_DIR = "01_paper_preprocess/output"
DEFAULT_CONCURRENCY = 50
DEFAULT_PDF_RENDER_DPI = 300
DEFAULT_PAGES_PER_REQUEST = 10
DEFAULT_IMAGE_MAX_MB = 4.0
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
    parser.add_argument(
        "--pages-per-request",
        type=int,
        default=None,
        help=(
            "Maximum PDF pages sent in one model request. Defaults to "
            f"PDF_IMAGES_PER_REQUEST, PDF_PAGES_PER_REQUEST, or {DEFAULT_PAGES_PER_REQUEST}."
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
        raise ValueError(f"{name} must be an integer, got: {raw_value}") from exc
    if value < 1:
        raise ValueError(f"{name} must be greater than or equal to 1, got: {value}")
    return value


def get_positive_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got: {raw_value}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0, got: {value}")
    return value


def find_pdf_files(paper_dir: Path) -> list[Path]:
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")
    if not paper_dir.is_dir():
        raise NotADirectoryError(f"Paper path is not a directory: {paper_dir}")
    return sorted(path for path in paper_dir.rglob("*.pdf") if path.is_file())


def iter_pdf_image_batches(
    path: Path,
    pages_per_request: int,
    max_image_bytes: int,
    uploader: QiniuUploader,
) -> Iterator[tuple[int, int, int, list[dict[str, Any]]]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: PyMuPDF. Install it with 'pip install PyMuPDF'."
        ) from exc

    zoom = DEFAULT_PDF_RENDER_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(path) as document:
        if document.page_count < 1:
            raise RuntimeError(f"PDF has no pages: {path}")

        batch_count = (document.page_count + pages_per_request - 1) // pages_per_request
        for batch_index, start_index in enumerate(
            range(0, document.page_count, pages_per_request)
        ):
            end_index = min(start_index + pages_per_request, document.page_count)
            image_parts: list[dict[str, Any]] = []
            for page_index in range(start_index, end_index):
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_bytes = pixmap.tobytes(IMAGE_FORMAT)
                page_number = page_index + 1
                if len(image_bytes) > max_image_bytes:
                    actual_mb = len(image_bytes) / (1024 * 1024)
                    limit_mb = max_image_bytes / (1024 * 1024)
                    raise RuntimeError(
                        f"Page {page_number} image is {actual_mb:.2f} MB, exceeding "
                        f"PDF_IMAGE_MAX_MB={limit_mb:.2f} MB."
                    )
                image_url = uploader.upload_image(
                    image_bytes,
                    object_key_for_pdf_page(str(path.resolve()), page_number),
                )
                image_parts.append(
                    {
                        "type": "input_image",
                        "image_url": image_url,
                    }
                )
            yield batch_index + 1, batch_count, start_index + 1, image_parts


def request_extraction(
    image_parts: list[dict[str, Any]],
    client: ResponsesClient,
    batch_index: int,
    batch_count: int,
    start_page: int,
) -> str:
    end_page = start_page + len(image_parts) - 1
    batch_note = (
        f"这是由一个 PDF 切分出的第 {batch_index}/{batch_count} 份，包含原 PDF "
        f"第 {start_page}-{end_page} 页。请将本次提供的页面独立视为一篇文献进行提取，"
        "只使用当前页面中明确出现的信息，不要补全其他分页中的内容。"
    )
    return client.respond(
        [
            {
                "role": "user",
                "content": [
                    *image_parts,
                    {"type": "input_text", "text": batch_note},
                    {"type": "input_text", "text": EXTRACTION_PROMPT},
                ],
            },
        ],
        reasoning_effort="high",
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
    client: ResponsesClient,
    pages_per_request: int,
    max_image_bytes: int,
    uploader: QiniuUploader,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "success",
        "data": [],
        "errors": [],
    }

    try:
        for batch_index, batch_count, start_page, image_parts in iter_pdf_image_batches(
            path, pages_per_request, max_image_bytes, uploader
        ):
            try:
                response_text = request_extraction(
                    image_parts,
                    client,
                    batch_index,
                    batch_count,
                    start_page,
                )
                extracted = parse_json_from_text(response_text)
                if not isinstance(extracted, dict):
                    raise RuntimeError(
                        "Model returned JSON, but the top-level value is not an object."
                    )
                result["data"].append(extracted)
            except Exception as exc:
                result["errors"].append(
                    f"part {batch_index}/{batch_count} (page {start_page}-"
                    f"{start_page + len(image_parts) - 1}): {exc}"
                )

        if result["errors"]:
            result["status"] = "partial" if result["data"] else "failed"
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append(str(exc))
        return result


def write_output(output_path: Path, results: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    client = create_client()
    uploader = create_qiniu_uploader()
    concurrency = get_int_env("MODEL_CONCURRENCY", DEFAULT_CONCURRENCY)
    pages_per_request = (
        args.pages_per_request
        if args.pages_per_request is not None
        else get_int_env(
            "PDF_IMAGES_PER_REQUEST",
            get_int_env("PDF_PAGES_PER_REQUEST", DEFAULT_PAGES_PER_REQUEST),
        )
    )
    if pages_per_request < 1:
        raise ValueError(
            f"--pages-per-request must be greater than or equal to 1, got: {pages_per_request}"
        )
    max_image_mb = get_positive_float_env("PDF_IMAGE_MAX_MB", DEFAULT_IMAGE_MAX_MB)
    max_image_bytes = int(max_image_mb * 1024 * 1024)
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
        f"images per request: {pages_per_request}; image size limit: {max_image_mb:g} MB"
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
    print("Rendering PDF pages, uploading them to Qiniu, and parsing PDFs...")
    results_by_path: dict[Path, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                process_pdf,
                path,
                client,
                pages_per_request,
                max_image_bytes,
                uploader,
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
            data = result.get("data")
            if isinstance(data, list) and data:
                results_by_path[path] = data
            if status == "success":
                tqdm.write(f"[success] {relative_path}: {len(data)} part(s)")
            elif status == "partial":
                tqdm.write(
                    f"[partial] {relative_path}: {len(data)} part(s) succeeded; "
                    + "; ".join(result.get("errors", []))
                )
            else:
                errors = result.get("errors", [])
                tqdm.write(
                    f"[failed] {relative_path}: "
                    + ("; ".join(errors) if errors else "unknown error")
                )

    results = [
        paper
        for path in pdf_files
        for paper in results_by_path.get(path, [])
    ]
    write_output(output_path, results)
    failed_count = len(pdf_files) - len(results_by_path)
    elapsed = time.perf_counter() - started_at
    print("Extraction finished.")
    print(f"Extracted document parts: {len(results)}")
    print(f"PDFs with no successful parts: {failed_count}")
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

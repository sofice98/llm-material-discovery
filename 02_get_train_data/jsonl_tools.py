# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import ResponsesClient, create_client
from llm_client.logging_utils import configure_script_logging


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_TRANSLATE_FIELDS = (
    "instruction",
    "input",
    "output",
    "text",
    "prompt",
    "chosen",
    "rejected",
)
LANGUAGES = {"zh": "Simplified Chinese", "en": "English"}


class JsonlError(RuntimeError):
    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate, merge, inspect, and count JSONL training data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    translate = subparsers.add_parser(
        "translate", help="Translate common text fields in a JSONL file."
    )
    translate.add_argument("input", type=Path, help="Source JSONL file.")
    translate.add_argument(
        "--language",
        required=True,
        choices=tuple(LANGUAGES),
        help="Target language: zh (Chinese) or en (English).",
    )
    translate.add_argument(
        "--batch-size", type=positive_int, default=10, help="Records per API request."
    )
    translate.add_argument(
        "--concurrency",
        type=positive_int,
        default=10,
        help="Concurrent API requests (default: 10).",
    )
    translate.add_argument(
        "--max-batch-chars",
        type=positive_int,
        default=30000,
        help="Maximum serialized source characters in one API request.",
    )

    merge = subparsers.add_parser("merge", help="Merge multiple JSONL files in order.")
    merge.add_argument("inputs", type=Path, nargs="+", help="Input JSONL files.")
    merge.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file. Defaults to 02_get_train_data/output/merged_<timestamp>.jsonl.",
    )

    stats = subparsers.add_parser(
        "task-stats", help="Show task_type counts and proportions for SFT JSONL."
    )
    stats.add_argument("input", type=Path, help="SFT JSONL file.")

    count = subparsers.add_parser(
        "count", help="Count valid records in a JSONL file or items in a JSON array file."
    )
    count.add_argument(
        "input", type=Path, help="JSONL file or JSON file with a top-level array."
    )

    tokens = subparsers.add_parser(
        "tokens", help="Count JSONL tokens with a tiktoken tokenizer."
    )
    tokens.add_argument("input", type=Path, help="JSONL file.")
    tokenizer = tokens.add_mutually_exclusive_group()
    tokenizer.add_argument("--model", help="Use tiktoken encoding_for_model(model).")
    tokenizer.add_argument(
        "--encoding", default="cl100k_base", help="tiktoken encoding name."
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def require_input(path: Path) -> Path:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    return resolved


def prepare_output(path: Path) -> Path:
    resolved = resolve_path(path)
    if resolved.exists():
        raise FileExistsError(f"Output already exists; choose a new output path: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def timestamped_output(prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return DEFAULT_OUTPUT_DIR / f"{prefix}_{timestamp}.jsonl"


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise JsonlError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, record


def parse_json_response(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
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


def make_batches(
    records: list[dict[str, Any]], fields: list[str], batch_size: int, max_chars: int
) -> list[list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        selected = {
            field: record[field]
            for field in fields
            if isinstance(record.get(field), str) and record[field].strip()
        }
        payloads.append({"id": index, "fields": selected})

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for payload in payloads:
        payload_chars = len(json.dumps(payload, ensure_ascii=False))
        if current and (len(current) >= batch_size or current_chars + payload_chars > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(payload)
        current_chars += payload_chars
    if current:
        batches.append(current)
    return batches


def translate_batch(
    batch: list[dict[str, Any]],
    language: str,
    client: ResponsesClient,
) -> dict[int, dict[str, str]]:
    source_json = json.dumps(batch, ensure_ascii=False, separators=(",", ":"))
    prompt = (
        f"Translate every string value inside each item's fields object into "
        f"{LANGUAGES[language]}. Preserve formulas, numbers, units, JSON structure, and meaning. "
        "Do not translate or change id values or field names. Return only a JSON array with "
        "exactly the same shape and item ids; do not add Markdown or explanations.\n\n"
        f"Source JSON:\n{source_json}"
    )
    response_text = client.respond(
        [{"role": "user", "content": prompt}],
        reasoning_effort="none",
    )
    parsed = parse_json_response(response_text)
    if not isinstance(parsed, list):
        raise RuntimeError("Translation response must be a JSON array.")

    expected = {item["id"]: set(item["fields"]) for item in batch}
    translated: dict[int, dict[str, str]] = {}
    for item in parsed:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise RuntimeError("Translation response contains an invalid item or id.")
        item_id = item["id"]
        fields = item.get("fields")
        if item_id not in expected or not isinstance(fields, dict):
            raise RuntimeError(f"Translation response contains unexpected id: {item_id}")
        if set(fields) != expected[item_id] or not all(
            isinstance(value, str) for value in fields.values()
        ):
            raise RuntimeError(f"Translation response fields do not match id {item_id}.")
        translated[item_id] = fields
    if set(translated) != set(expected):
        raise RuntimeError("Translation response is missing one or more records.")
    return translated


def command_translate(args: argparse.Namespace) -> None:
    input_path = require_input(args.input)
    output_path = prepare_output(
        timestamped_output(f"{input_path.stem}_{args.language}")
    )
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    records = [record for _, record in iter_jsonl(input_path)]
    batches = make_batches(
        records, list(DEFAULT_TRANSLATE_FIELDS), args.batch_size, args.max_batch_chars
    )
    client = create_client()
    results: dict[int, dict[str, str]] = {}
    failed_batches: list[tuple[list[dict[str, Any]], str]] = []

    print(
        f"Translating {len(records)} records in {len(batches)} batches to "
        f"{LANGUAGES[args.language]} with {client.model}..."
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_batch = {
            executor.submit(translate_batch, batch, args.language, client): batch
            for batch in batches
        }
        for future in tqdm(
            as_completed(future_to_batch), total=len(future_to_batch), desc="Translating"
        ):
            batch = future_to_batch[future]
            try:
                translated = future.result()
                results.update(translated)
                tqdm.write(
                    f"[success] Samples {batch[0]['id']} to {batch[-1]['id']}: "
                    f"translated {len(translated)} samples"
                )
            except Exception as exc:
                failed_batches.append((batch, str(exc)))
                tqdm.write(
                    f"[failed] Samples {batch[0]['id']} to {batch[-1]['id']}: {exc}"
                )

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for index, record in enumerate(records):
            if index not in results:
                continue
            record.update(results[index])
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(results)} translated records to {output_path}")
    print(f"Successful requests: {len(batches) - len(failed_batches)}")
    print(f"Failed requests: {len(failed_batches)}")
    print(f"Translated samples: {len(results)}")
    print(f"Untranslated samples: {sum(len(batch) for batch, _ in failed_batches)}")


def command_merge(args: argparse.Namespace) -> None:
    input_paths = [require_input(path) for path in args.inputs]
    output_path = prepare_output(args.output or timestamped_output("merged"))
    if output_path in input_paths:
        raise ValueError("The output path cannot also be an input path.")
    written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for input_path in input_paths:
            for _, record in iter_jsonl(input_path):
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                output.write(line + "\n")
                written += 1
    print(f"Merged {len(input_paths)} files and wrote {written} records to {output_path}")


def command_task_stats(args: argparse.Namespace) -> None:
    input_path = require_input(args.input)
    counts: Counter[str] = Counter()
    missing = 0
    total = 0
    for _, record in iter_jsonl(input_path):
        total += 1
        task_type = record.get("task_type")
        if isinstance(task_type, str) and task_type.strip():
            counts[task_type.strip()] += 1
        else:
            missing += 1
    print(f"File: {input_path}")
    print(f"Total records: {total}")
    print(f"Records without task_type: {missing}")
    print(f"{'task_type':<45} {'count':>10} {'proportion':>12}")
    print("-" * 69)
    for task_type, count in counts.most_common():
        proportion = count / total if total else 0.0
        print(f"{task_type:<45} {count:>10} {proportion:>11.2%}")


def command_count(args: argparse.Namespace) -> None:
    input_path = require_input(args.input)
    if input_path.suffix.lower() == ".json":
        try:
            with input_path.open("r", encoding="utf-8-sig") as file:
                records = json.load(file)
        except json.JSONDecodeError as exc:
            raise JsonlError(f"Invalid JSON in {input_path}: {exc}") from exc
        if not isinstance(records, list):
            raise JsonlError(
                f"Expected a top-level JSON array in {input_path}; use a JSONL file for line-delimited records."
            )
        sample_count = len(records)
    else:
        sample_count = sum(1 for _ in iter_jsonl(input_path))
    print(f"File: {input_path}")
    print(f"Samples: {sample_count}")


def command_tokens(args: argparse.Namespace) -> None:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("tiktoken is required; run: pip install -r requirements.txt") from exc

    input_path = require_input(args.input)
    encoding = (
        tiktoken.encoding_for_model(args.model)
        if args.model
        else tiktoken.get_encoding(args.encoding)
    )
    counts: list[int] = []
    for _, record in tqdm(iter_jsonl(input_path), desc="Counting tokens"):
        text = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        counts.append(len(encoding.encode(text)))
    total = sum(counts)
    print(f"File: {input_path}")
    print(f"Tokenizer: {encoding.name}")
    print(f"Records: {len(counts)}")
    print(f"Total tokens: {total}")
    print(f"Average tokens/record: {total / len(counts):.2f}" if counts else "Average tokens/record: 0")
    print(f"Minimum tokens/record: {min(counts) if counts else 0}")
    print(f"Maximum tokens/record: {max(counts) if counts else 0}")


def main() -> int:
    args = parse_args()
    commands = {
        "translate": command_translate,
        "merge": command_merge,
        "task-stats": command_task_stats,
        "count": command_count,
        "tokens": command_tokens,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    configure_script_logging(__file__)
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, JsonlError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)

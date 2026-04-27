#!/usr/bin/env python3
"""Summarize lm-eval result JSON files into a small table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


PREFERRED_METRICS = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "f1,none",
    "rougeL,none",
    "bleu,none",
    "word_perplexity,none",
    "perplexity,none",
    "brier_score,none",
)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def iter_result_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("results*.json"))
    return sorted(set(files))


def run_name_for(file_path: Path, roots: list[Path]) -> str:
    parent = file_path.parent
    for root in roots:
        if root.is_dir():
            try:
                rel = parent.relative_to(root)
                return str(rel) if str(rel) != "." else parent.name
            except ValueError:
                pass
    return parent.name


def selected_metrics(metrics: dict[str, object], all_metrics: bool) -> list[tuple[str, float]]:
    if all_metrics:
        return [
            (key, float(value))
            for key, value in sorted(metrics.items())
            if is_number(value) and "stderr" not in key
        ]

    for key in PREFERRED_METRICS:
        value = metrics.get(key)
        if is_number(value):
            return [(key, float(value))]

    for key, value in sorted(metrics.items()):
        if is_number(value) and "stderr" not in key:
            return [(key, float(value))]

    return []


def load_rows(files: list[Path], roots: list[Path], all_metrics: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for file_path in files:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: failed to read {file_path}: {exc}", file=sys.stderr)
            continue

        results = data.get("results", {})
        if not isinstance(results, dict):
            continue

        run_name = run_name_for(file_path, roots)
        for task_name, metrics in sorted(results.items()):
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in selected_metrics(metrics, all_metrics):
                rows.append(
                    {
                        "run": run_name,
                        "task": task_name,
                        "metric": metric_name,
                        "value": f"{value:.8g}",
                        "file": str(file_path),
                    }
                )
    return rows


def print_markdown(rows: list[dict[str, str]]) -> None:
    print("| run | task | metric | value | file |")
    print("| --- | --- | --- | ---: | --- |")
    for row in rows:
        print(
            f"| {row['run']} | {row['task']} | {row['metric']} | "
            f"{row['value']} | {row['file']} |"
        )


def print_csv(rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=["run", "task", "metric", "value", "file"])
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Result files or directories")
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output table format",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Emit all numeric non-stderr metrics instead of one preferred metric per task",
    )
    args = parser.parse_args()

    files = iter_result_files(args.paths)
    if not files:
        print("no lm-eval results*.json files found", file=sys.stderr)
        return 1

    rows = load_rows(files, args.paths, args.all_metrics)
    if not rows:
        print("no numeric lm-eval task metrics found", file=sys.stderr)
        return 1

    if args.format == "csv":
        print_csv(rows)
    else:
        print_markdown(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

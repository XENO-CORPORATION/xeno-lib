#!/usr/bin/env python3
"""Generate and gate an objective FFmpeg parity matrix for xeno-lib/xeno-edit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


STATUS_RANK = {"missing": 0, "partial": 1, "have": 2}
STATUS_ICON = {"missing": "MISSING", "partial": "PARTIAL", "have": "HAVE"}


def run_cmd(args: List[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        cmd = " ".join(args)
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{stdout}\n{stderr}")
    return stdout + stderr


def parse_json_object(raw: str) -> Dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in command output.")
    return json.loads(raw[start : end + 1])


def ffmpeg_counts(ffmpeg_bin: str) -> Dict[str, int]:
    def count(pattern: str, subcommand: str) -> int:
        text = run_cmd([ffmpeg_bin, "-hide_banner", subcommand])
        return len(re.findall(pattern, text, flags=re.MULTILINE))

    return {
        "encoders": count(r"^\s+[VAS]\S{5}\s+\S+", "-encoders"),
        "decoders": count(r"^\s+[VAS]\S{5}\s+\S+", "-decoders"),
        "muxers": count(r"^\s+[E\.]\s+\S+", "-muxers"),
        "demuxers": count(r"^\s+[D\.]\s+\S+", "-demuxers"),
        "filters": count(r"^\s+[T\.][S\.]\s+\S+", "-filters"),
    }


def get_help_text(xeno_bin: str, command: str | None = None) -> str:
    args = [xeno_bin]
    if command:
        args.append(command)
    args.append("--help")
    return run_cmd(args)


def evaluate_entry(
    entry: Dict[str, Any],
    capabilities: Dict[str, Any],
    root_help: str,
    command_help_cache: Dict[str, str],
) -> Dict[str, Any]:
    status_present = entry.get("status_if_present", "have")
    status_absent = entry.get("status_if_absent", "missing")
    etype = entry["type"]

    present = False
    evidence = ""

    if etype == "capability_contains":
        key = entry["capability"]
        value = entry["value"]
        values = capabilities.get(key, [])
        present = isinstance(values, list) and value in values
        evidence = f"capabilities.{key} contains '{value}'" if present else f"capabilities.{key} missing '{value}'"
    elif etype == "capability_bool":
        key = entry["capability"]
        present = bool(capabilities.get(key, False))
        evidence = f"capabilities.{key}={present}"
    elif etype == "command_exists":
        command = entry["command"]
        present = re.search(rf"^\s{{2,}}{re.escape(command)}\s{{2,}}", root_help, re.MULTILINE) is not None
        evidence = f"command '{command}' present in root help" if present else f"command '{command}' missing in root help"
    elif etype == "command_help_contains":
        command = entry["command"]
        needle = entry["contains"]
        help_text = command_help_cache.get(command)
        if help_text is None:
            help_text = get_help_text(entry["_xeno_bin"], command)
            command_help_cache[command] = help_text
        present = needle in help_text
        evidence = f"{command} --help contains '{needle}'" if present else f"{command} --help missing '{needle}'"
    elif etype == "always":
        present = bool(entry.get("present", False))
        evidence = entry.get("evidence", "manual classification")
    else:
        raise ValueError(f"Unsupported entry type: {etype}")

    status = status_present if present else status_absent
    if status not in STATUS_RANK:
        raise ValueError(f"Invalid status '{status}' in entry {entry['id']}")

    return {
        "id": entry["id"],
        "category": entry["category"],
        "capability": entry["capability_name"],
        "ffmpeg_reference": entry["ffmpeg_reference"],
        "xeno_reference": entry["xeno_reference"],
        "status": status,
        "evidence": evidence,
        "notes": entry.get("notes", ""),
    }


def weighted_score(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        if row["status"] == "have":
            total += 1.0
        elif row["status"] == "partial":
            total += 0.5
    return (total / len(rows)) * 100.0


def build_markdown(
    report: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# FFmpeg Parity Matrix (Generated)")
    lines.append("")
    lines.append(f"- Generated (UTC): `{report['generated_at_utc']}`")
    lines.append(f"- xeno binary: `{report['xeno_bin']}`")
    lines.append(f"- ffmpeg binary: `{report['ffmpeg_bin']}`")
    lines.append("")
    lines.append("## FFmpeg Surface Snapshot")
    lines.append("")
    lines.append("| Encoders | Decoders | Muxers | Demuxers | Filters |")
    lines.append("|---:|---:|---:|---:|---:|")
    fc = report["ffmpeg_counts"]
    lines.append(f"| {fc['encoders']} | {fc['decoders']} | {fc['muxers']} | {fc['demuxers']} | {fc['filters']} |")
    lines.append("")
    summary = report["summary"]
    lines.append("## xeno-lib/xeno-edit Parity Summary")
    lines.append("")
    lines.append("| Total Rows | HAVE | PARTIAL | MISSING | Weighted Score |")
    lines.append("|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {summary['total_rows']} | {summary['have']} | {summary['partial']} | {summary['missing']} | {summary['weighted_score_pct']:.2f}% |"
    )
    lines.append("")

    rows_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in report["rows"]:
        rows_by_category[row["category"]].append(row)

    for category in report["category_order"]:
        rows = rows_by_category.get(category, [])
        if not rows:
            continue
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Capability | FFmpeg | xeno-lib/xeno-edit | Status | Evidence | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for row in rows:
            lines.append(
                f"| {row['capability']} | {row['ffmpeg_reference']} | {row['xeno_reference']} | "
                f"{STATUS_ICON[row['status']]} | {row['evidence']} | {row['notes']} |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FFmpeg parity matrix for xeno-lib/xeno-edit.")
    parser.add_argument("--xeno-bin", required=True, help="Path to xeno-edit binary.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="Path to ffmpeg binary.")
    parser.add_argument("--spec", required=True, help="Path to parity spec JSON.")
    parser.add_argument("--output-json", required=True, help="Output path for generated parity report JSON.")
    parser.add_argument("--output-md", required=True, help="Output path for generated parity matrix markdown.")
    parser.add_argument("--baseline", help="Baseline status JSON for regression checks.")
    parser.add_argument("--baseline-candidate", help="Write current statuses to this baseline candidate JSON path.")
    parser.add_argument("--fail-on-regression", action="store_true", help="Fail if any status regresses vs baseline.")
    return parser.parse_args()


def compare_with_baseline(report: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    current = {row["id"]: row["status"] for row in report["rows"]}
    regressions: List[str] = []
    missing_ids: List[str] = []

    for fid, fstatus in baseline.get("statuses", {}).items():
        cstatus = current.get(fid)
        if cstatus is None:
            missing_ids.append(fid)
            continue
        if STATUS_RANK[cstatus] < STATUS_RANK[fstatus]:
            regressions.append(f"{fid}: {fstatus} -> {cstatus}")

    return regressions, missing_ids


def main() -> int:
    args = parse_args()
    spec_path = Path(args.spec)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    spec = load_json(spec_path)
    root_help = get_help_text(args.xeno_bin)
    capabilities = parse_json_object(run_cmd([args.xeno_bin, "capabilities"]))
    counts = ffmpeg_counts(args.ffmpeg_bin)

    command_help_cache: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []
    for entry in spec["entries"]:
        entry = dict(entry)
        entry["_xeno_bin"] = args.xeno_bin
        rows.append(evaluate_entry(entry, capabilities, root_help, command_help_cache))

    have = sum(1 for r in rows if r["status"] == "have")
    partial = sum(1 for r in rows if r["status"] == "partial")
    missing = sum(1 for r in rows if r["status"] == "missing")

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "xeno_bin": args.xeno_bin,
        "ffmpeg_bin": args.ffmpeg_bin,
        "ffmpeg_counts": counts,
        "summary": {
            "total_rows": len(rows),
            "have": have,
            "partial": partial,
            "missing": missing,
            "weighted_score_pct": weighted_score(rows),
        },
        "category_order": spec.get("category_order", []),
        "rows": rows,
    }

    write_json(output_json, report)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown(report), encoding="utf-8")

    if args.baseline_candidate:
        candidate = {
            "generated_at_utc": report["generated_at_utc"],
            "source_report": str(output_json),
            "statuses": {row["id"]: row["status"] for row in rows},
        }
        write_json(Path(args.baseline_candidate), candidate)

    if args.baseline:
        baseline = load_json(Path(args.baseline))
        regressions, missing_ids = compare_with_baseline(report, baseline)
        if regressions:
            print("Parity regressions detected:")
            for item in regressions:
                print(f"- {item}")
        if missing_ids:
            print("Baseline IDs missing from current report:")
            for item in missing_ids:
                print(f"- {item}")

        if args.fail_on_regression and (regressions or missing_ids):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

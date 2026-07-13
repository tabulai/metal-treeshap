#!/usr/bin/env python3
"""Parse powermetrics plist samples and attribute estimated power to timed suite jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import plistlib
import tempfile
from pathlib import Path


def load_samples(path: Path) -> list[dict]:
    """Load the NUL-separated plist dictionaries emitted by powermetrics."""
    samples: list[dict] = []
    for index, chunk in enumerate(path.read_bytes().split(b"\0")):
        if not chunk.strip():
            continue
        try:
            sample = plistlib.loads(chunk)
        except Exception as error:
            raise ValueError(
                f"invalid powermetrics plist sample {index}: {error}"
            ) from error
        if isinstance(sample, dict):
            samples.append(sample)
    return samples


def _timestamp(value) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=value.tzinfo or dt.timezone.utc).astimezone(
            dt.timezone.utc
        )
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                dt.timezone.utc
            )
        except ValueError:
            return None
    return None


def _job_window(job: dict) -> tuple[dt.datetime, dt.datetime]:
    start = _timestamp(job.get("started_utc"))
    end = _timestamp(job.get("finished_utc"))
    if start is None or end is None or end < start:
        raise ValueError("job has an invalid started_utc/finished_utc window")
    return start, end


def _job_windows(job: dict) -> list[tuple[dt.datetime, dt.datetime]]:
    """Return exact timed-call windows when available, else the aggregate window."""
    raw_windows = job.get("sample_windows_utc")
    if not isinstance(raw_windows, list) or not raw_windows:
        return [_job_window(job)]
    windows = [_job_window(window) for window in raw_windows]
    for previous, current in zip(windows, windows[1:]):
        if current[0] < previous[1]:
            raise ValueError("job sample_windows_utc overlap or are out of order")
    return windows


def summarize_jobs(samples: list[dict], jobs: list[dict]) -> list[dict]:
    """Estimate per-job average power and energy from overlapping sample intervals.

    ``cpu_power`` and ``gpu_power`` are documented by powermetrics' text output in mW.
    A sample covers ``[timestamp - elapsed_ns, timestamp]``. Energy is integrated only
    over the intersection of that interval and a job's wall-clock window.
    """
    parsed: list[dict] = []
    for sample in samples:
        end = _timestamp(sample.get("timestamp"))
        try:
            elapsed_s = float(sample.get("elapsed_ns", 0)) / 1e9
        except (TypeError, ValueError):
            continue
        if end is None or elapsed_s <= 0:
            continue
        processor = sample.get("processor")
        processor = processor if isinstance(processor, dict) else {}
        parsed.append(
            {
                "start": end - dt.timedelta(seconds=elapsed_s),
                "end": end,
                "gpu_mw": processor.get("gpu_power"),
                "cpu_mw": processor.get("cpu_power"),
                "thermal": sample.get("thermal_pressure"),
            }
        )

    summaries: list[dict] = []
    for job in jobs:
        windows = _job_windows(job)
        metric = {
            "gpu": {"weighted_mw_s": 0.0, "covered_s": 0.0, "samples": 0},
            "cpu": {"weighted_mw_s": 0.0, "covered_s": 0.0, "samples": 0},
        }
        overlapping = 0
        thermal: set[str] = set()
        for sample in parsed:
            overlap = sum(
                max(
                    0.0,
                    (min(end, sample["end"]) - max(start, sample["start"]))
                    .total_seconds(),
                )
                for start, end in windows
            )
            if overlap <= 0:
                continue
            overlapping += 1
            if sample["thermal"] is not None:
                thermal.add(str(sample["thermal"]))
            for name in ("gpu", "cpu"):
                try:
                    power_mw = float(sample[f"{name}_mw"])
                except (TypeError, ValueError):
                    continue
                metric[name]["weighted_mw_s"] += power_mw * overlap
                metric[name]["covered_s"] += overlap
                metric[name]["samples"] += 1

        summary: dict = {
            "status": "ok" if overlapping else "unavailable",
            "overlapping_samples": overlapping,
            "job_duration_s": sum((end - start).total_seconds()
                                  for start, end in windows),
            "job_window_count": len(windows),
            "thermal_pressure_levels": sorted(thermal),
            "provenance": {
                "source": "powermetrics plist cpu_power,gpu_power,thermal samplers",
                "power_unit": "mW",
                "energy_method": "integral of sampled power over job/sample overlap",
                "caveat": (
                    "Apple describes these power values as estimates; use within-device "
                    "optimization comparisons, not cross-device claims."
                ),
            },
        }
        identity = {
            key: job[key]
            for key in (
                "sequence", "outer_round", "workload_name", "rows",
                "dataset", "size", "device", "iteration", "order_index",
                "block_round", "calls", "power_block", "explained_rows",
            )
            if key in job
        }
        if identity:
            summary["job"] = identity
        for name in ("gpu", "cpu"):
            covered = metric[name]["covered_s"]
            if covered:
                summary[f"{name}_mean_mw"] = metric[name]["weighted_mw_s"] / covered
                summary[f"{name}_estimated_energy_j"] = (
                    metric[name]["weighted_mw_s"] / 1000.0
                )
                explained_rows = job.get("explained_rows")
                if isinstance(explained_rows, int) and explained_rows > 0:
                    summary[f"{name}_estimated_energy_j_per_explained_row"] = (
                        summary[f"{name}_estimated_energy_j"] / explained_rows
                    )
                summary[f"{name}_covered_s"] = covered
                summary[f"{name}_samples"] = metric[name]["samples"]
            else:
                summary[f"{name}_missing_reason"] = (
                    "no overlapping sample contained power"
                )
        if not overlapping:
            summary["missing_reason"] = (
                "no valid powermetrics interval overlaps this job"
            )
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("powermetrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    jobs = suite.get("results")
    if not isinstance(jobs, list):
        raise SystemExit("suite contains no results array")
    artifact = {
        "schema": "metal_treeshap.phase2.power_summary.v1",
        "suite": str(args.suite.resolve()),
        "powermetrics": str(args.powermetrics.resolve()),
        "jobs": summarize_jobs(load_samples(args.powermetrics), jobs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=args.output.parent,
        delete=False,
        prefix=args.output.name + ".",
        suffix=".tmp",
    ) as temp:
        json.dump(artifact, temp, indent=2, sort_keys=True)
        temp.write("\n")
        temporary = Path(temp.name)
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()

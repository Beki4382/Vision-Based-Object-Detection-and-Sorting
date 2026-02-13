#!/usr/bin/env python3

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetricsSummary:
    path: Path
    frames: int
    avg_ms: float | None
    min_ms: float | None
    max_ms: float | None
    std_ms: float | None
    avg_dets: float | None
    pct_frames_with_det: float | None


def summarize_csv(path: Path) -> MetricsSummary:
    if not path.exists():
        return MetricsSummary(
            path=path,
            frames=0,
            avg_ms=None,
            min_ms=None,
            max_ms=None,
            std_ms=None,
            avg_dets=None,
            pct_frames_with_det=None,
        )

    times: list[float] = []
    dets: list[int] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expected header: frame_id,inference_time_ms,num_detections,timestamp
            try:
                times.append(float(row["inference_time_ms"]))
                dets.append(int(float(row["num_detections"])))
            except Exception:
                continue

    if not times:
        return MetricsSummary(
            path=path,
            frames=0,
            avg_ms=None,
            min_ms=None,
            max_ms=None,
            std_ms=None,
            avg_dets=None,
            pct_frames_with_det=None,
        )

    frames = len(times)
    avg_ms = statistics.mean(times)
    min_ms = min(times)
    max_ms = max(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    avg_dets = statistics.mean(dets) if dets else 0.0
    frames_with_det = sum(1 for d in dets if d > 0)
    pct_frames_with_det = 100.0 * frames_with_det / frames if frames > 0 else 0.0

    return MetricsSummary(
        path=path,
        frames=frames,
        avg_ms=avg_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        std_ms=std_ms,
        avg_dets=avg_dets,
        pct_frames_with_det=pct_frames_with_det,
    )


def format_summary(name: str, s: MetricsSummary) -> str:
    if s.frames == 0:
        return f"- **{name}**: no frames recorded (missing or empty CSV: `{s.path}`)"

    return "\n".join(
        [
            f"- **{name}** (`{s.path.name}`):",
            f"  - Frames: {s.frames}",
            f"  - Avg inference: {s.avg_ms:.2f} ms (min {s.min_ms:.2f}, max {s.max_ms:.2f}, std {s.std_ms:.2f})",
            f"  - Avg detections/frame: {s.avg_dets:.2f}",
            f"  - Frames with ≥1 detection: {s.pct_frames_with_det:.1f}%",
        ]
    )


def main() -> None:
    base = Path("/home/beki/Vision-Based-Object-Detection-and-Sorting/comparison/robustness_test/runs_low_light")
    yolo_csv = base / "yolo_low_light_metrics.csv"
    rtdetr_csv = base / "rtdetr_low_light_metrics.csv"

    yolo = summarize_csv(yolo_csv)
    rtdetr = summarize_csv(rtdetr_csv)

    out_md = base / "LOW_LIGHT_METRICS_SUMMARY.md"
    out_md.write_text(
        "\n".join(
            [
                "## Low-light robustness (scene-only) — metrics summary",
                "",
                "These metrics come from running `scene.launch.py` only (vision + scene manager) in a dim-light Gazebo world for ~90 seconds.",
                "",
                format_summary("YOLOv11", yolo),
                "",
                format_summary("RT-DETR", rtdetr),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()


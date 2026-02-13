#!/usr/bin/env python3

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CubeResult:
    cube_key: str
    target_xy: Tuple[float, float]
    final_xy: Optional[Tuple[float, float]]
    dist_m: Optional[float]
    success: bool


POSE_BLOCK_RE = re.compile(r"CUBE POSITIONS:", re.IGNORECASE)
POSE_LINE_RE = re.compile(
    r"^\s*(?P<name>BIG_GREEN|GREEN|BIG_RED|RED)\s*:.*robot=\(\s*(?P<x>-?\d+\.\d+)\s*,\s*(?P<y>-?\d+\.\d+)\s*\)",
    re.IGNORECASE,
)


def load_place_positions(scene_manager_path: Path) -> Dict[str, Tuple[float, float]]:
    """
    Extract place_positions = { 'big_green': {'x': ..., 'y': ...}, ... }
    """
    txt = scene_manager_path.read_text(encoding="utf-8", errors="ignore")
    # Narrow to the dict body for robustness
    m = re.search(r"place_positions\s*=\s*\{(?P<body>[\s\S]*?)\}\s*\n\s*#\s*Priority", txt)
    if not m:
        raise RuntimeError(f"Could not find place_positions in {scene_manager_path}")
    body = m.group("body")

    item_re = re.compile(
        r"'(?P<key>big_green|small_green|big_red|small_red)'\s*:\s*\{\s*'x'\s*:\s*(?P<x>-?\d+\.\d+)\s*,\s*'y'\s*:\s*(?P<y>-?\d+\.\d+)\s*\}",
        re.IGNORECASE,
    )
    out: Dict[str, Tuple[float, float]] = {}
    for im in item_re.finditer(body):
        out[im.group("key").lower()] = (float(im.group("x")), float(im.group("y")))
    if len(out) != 4:
        raise RuntimeError(f"Expected 4 place positions, got {len(out)} from {scene_manager_path}")
    return out


def parse_last_pose_block(scene_log: Path) -> Dict[str, Tuple[float, float]]:
    """
    Returns robot-frame XY from the LAST pose_logger block.
    Keys returned: BIG_GREEN, GREEN, BIG_RED, RED (upper).
    """
    lines = scene_log.read_text(encoding="utf-8", errors="ignore").splitlines()
    # Find last "CUBE POSITIONS:" line index
    last_idx = None
    for idx, line in enumerate(lines):
        if POSE_BLOCK_RE.search(line):
            last_idx = idx
    if last_idx is None:
        return {}

    found: Dict[str, Tuple[float, float]] = {}
    # Parse next ~10 lines
    for line in lines[last_idx : last_idx + 12]:
        lm = POSE_LINE_RE.match(line)
        if not lm:
            continue
        name = lm.group("name").upper()
        found[name] = (float(lm.group("x")), float(lm.group("y")))
    return found


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def evaluate_trial(
    scene_log: Path,
    place_positions: Dict[str, Tuple[float, float]],
    tolerance_m: float,
) -> List[CubeResult]:
    final_xy_raw = parse_last_pose_block(scene_log)
    # Map pose_logger names to our cube keys
    map_keys = {
        "BIG_GREEN": "big_green",
        "GREEN": "small_green",
        "BIG_RED": "big_red",
        "RED": "small_red",
    }
    results: List[CubeResult] = []
    for pose_name, cube_key in map_keys.items():
        target = place_positions[cube_key]
        final_xy = final_xy_raw.get(pose_name)
        if final_xy is None:
            results.append(CubeResult(cube_key, target, None, None, False))
            continue
        d = dist(final_xy, target)
        results.append(CubeResult(cube_key, target, final_xy, d, d <= tolerance_m))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/beki/Vision-Based-Object-Detection-and-Sorting/comparison")
    ap.add_argument("--model", choices=["yolo", "rtdetr"], required=True)
    ap.add_argument("--tolerance", type=float, default=0.06, help="Success radius in meters (robot frame)")
    ap.add_argument("--attempts", type=int, default=10, help="Number of pick actions to score")
    args = ap.parse_args()

    root = Path(args.root)
    runs_dir = root / "accuracy_test" / "runs" / args.model
    if args.model == "yolo":
        scene_manager = root / "Comp_perfect_size_v3" / "ros2_ws" / "src" / "simpler_pick_place" / "scripts" / "scene_manager.py"
    else:
        scene_manager = root / "Comp_perfect_RT_v1" / "ros2_ws" / "src" / "simpler_pick_place" / "scripts" / "scene_manager.py"

    place_positions = load_place_positions(scene_manager)

    # Trials in order
    trials = sorted([p for p in runs_dir.glob("trial_*") if p.is_dir()], key=lambda p: p.name)
    if not trials:
        raise SystemExit(f"No trials found under {runs_dir}")

    # Attempt sequence per trial (priority order)
    attempt_order = ["big_green", "small_green", "big_red", "small_red"]

    attempt_results: List[CubeResult] = []
    per_trial_summary = []

    for t in trials:
        scene_log = t / "scene.log"
        res = evaluate_trial(scene_log, place_positions, args.tolerance)
        by_key = {r.cube_key: r for r in res}
        per_trial_summary.append((t.name, res))
        for k in attempt_order:
            attempt_results.append(by_key[k])

    scored = attempt_results[: args.attempts]
    successes = sum(1 for r in scored if r.success)
    total = len(scored)

    # Write markdown report
    out_md = root / f"ACCURACY_REPORT_{args.model.upper()}.md"
    lines: List[str] = []
    lines.append(f"# Accuracy Report ({args.model.upper()})")
    lines.append("")
    lines.append("## Definition of accuracy")
    lines.append(f"- **Unit of attempt**: one cube pick-and-place action")
    lines.append(f"- **Success condition**: after the controller finishes, the cube's *ground-truth* final position (from `pose_logger`) is within **{args.tolerance:.2f} m** of its target bin center (from `scene_manager.place_positions`) in **robot frame (x,y)**.")
    lines.append(f"- **Attempts scored**: first **{args.attempts}** attempts across trials (each trial produces 4 attempts in fixed order: big_green, small_green, big_red, small_red).")
    lines.append("")
    lines.append("## Results")
    lines.append(f"- **Successes**: **{successes}/{total}**")
    lines.append(f"- **Accuracy**: **{(successes/total*100.0 if total else 0):.1f}%**")
    lines.append("")
    lines.append("## Per-attempt details (scored set)")
    lines.append("| Attempt | Cube | Target (x,y) | Final (x,y) | Distance (m) | Success |")
    lines.append("|---:|---|---|---|---:|:---:|")
    for idx, r in enumerate(scored, start=1):
        tgt = f"({r.target_xy[0]:.2f},{r.target_xy[1]:.2f})"
        fin = "N/A" if r.final_xy is None else f"({r.final_xy[0]:.2f},{r.final_xy[1]:.2f})"
        dm = "N/A" if r.dist_m is None else f\"{r.dist_m:.3f}\"
        ok = "✓" if r.success else "✗"
        lines.append(f\"| {idx} | {r.cube_key} | {tgt} | {fin} | {dm} | {ok} |\")
    lines.append("")
    lines.append("## Trials used")
    for trial_name, _ in per_trial_summary:
        lines.append(f\"- `{runs_dir}/{trial_name}`\")
    lines.append("")
    lines.append("## Notes / limitations")
    lines.append("- This accuracy metric validates **end placement correctness** using Gazebo ground-truth poses. It does not separately score grasp-only success vs place-only success.")
    lines.append("- If a trial times out or the controller fails early, cubes typically remain near their initial positions and will be scored as failures (distance > tolerance).")
    lines.append("")
    out_md.write_text(\"\\n\".join(lines), encoding=\"utf-8\")
    print(f\"Wrote {out_md}\")


if __name__ == \"__main__\":\n+    main()


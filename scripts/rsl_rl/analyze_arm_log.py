"""Summarize actual arm motion from sim2sim_play.py --joint-log, without MuJoCo.

P10-P90 excludes isolated angle spikes; full and transition P2P are reported
separately. Reference-phase fitting checks the configured oscillator rather
than assuming that the policy follows it. No metric implies trial acceptance.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(path, warmup, period):
    with open(path, newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError(f"Empty joint log: {path}")
    names = records[0].keys()
    data = {name: np.array([float(row[name]) for row in records]) for name in names}
    time = data["t"]
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("The log must contain monotonically increasing timestamps")
    dt = float(np.median(np.diff(time)))
    commands = np.stack([data[k] for k in ("cmd_vx", "cmd_vy", "cmd_wz")], axis=1)
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(commands, axis=0) != 0, axis=1)) + 1]
    ends = np.r_[starts[1:], len(time)]
    result = {"source": str(Path(path).resolve()), "warmup_s": warmup, "reference_period_s": period,
              "fall_samples": int(data["fall"].sum()), "segments": []}
    for start, end in zip(starts, ends):
        segment_start = time[start] - dt
        indices = np.arange(start, end)
        steady = indices[time[indices] >= segment_start + warmup]
        transition = indices[time[indices] < segment_start + warmup]
        if len(steady) < 3:
            raise ValueError(f"Segment starting at {segment_start:g}s is shorter than the analysis window")
        ts = time[steady]
        design = np.stack([np.ones_like(ts), np.cos(2*np.pi*ts/period), np.sin(2*np.pi*ts/period)], axis=1)
        row = {
            "segment_s": [float(segment_start), float(time[end-1])],
            "steady_source_lines": [int(steady[0])+2, int(steady[-1])+2],
            "command": commands[start].tolist(),
            "actual_velocity_mean": [float(data[k][steady].mean()) for k in ("vx_yaw", "vy_yaw", "wz")],
            "pitch_std_deg": float(np.degrees(data["pitch"][steady]).std()),
            "roll_std_deg": float(np.degrees(data["roll"][steady]).std()),
            "height_p95_p05_m": float(np.ptp(np.percentile(data["base_z"][steady], [5, 95]))),
            "arms": {},
        }
        arm_signals = []
        for side in ("left", "right"):
            shoulder = data[f"{side}_shoulder_pitch_joint"]
            q = shoulder[steady]
            coeff = np.linalg.lstsq(design, q, rcond=None)[0]
            variance_sum = float(np.sum((q - q.mean())**2))
            contact = data[f"{side}_contact"]
            touchdowns = np.flatnonzero(np.diff(contact) > 0) + 1
            touchdowns = touchdowns[(touchdowns >= steady[0]) & (touchdowns <= steady[-1])]
            intervals = np.diff(time[touchdowns])
            arm_signals.append(q)
            row["arms"][side] = {
                "steady_p10_p90_deg": float(np.degrees(np.ptp(np.percentile(q, [10, 90])))),
                "steady_p2p_deg": float(np.degrees(np.ptp(q))),
                "full_segment_p2p_deg": float(np.degrees(np.ptp(shoulder[indices]))),
                "transition_p2p_deg": float(np.degrees(np.ptp(shoulder[transition]))) if len(transition) else None,
                "reference_amplitude_deg": float(np.degrees(np.hypot(coeff[1], coeff[2]))),
                "reference_explained_variance": float(1-np.sum((q-design@coeff)**2)/variance_sum) if variance_sum else None,
                "touchdown_times_s": time[touchdowns].tolist(),
                "stride_period_mean_s": float(intervals.mean()) if len(intervals) else None,
            }
        row["left_right_arm_correlation"] = (
            float(np.corrcoef(*arm_signals)[0, 1])
            if all(np.std(signal) > 0 for signal in arm_signals) else None
        )
        result["segments"].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("joint_logs", nargs="+")
    parser.add_argument("--warmup", type=float, default=1.5, help="Seconds excluded at each command transition.")
    parser.add_argument("--period", type=float, default=1.0, help="Configured arm target period in seconds.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()
    if args.warmup < 0 or args.period <= 0:
        parser.error("warmup must be non-negative and period must be positive")
    results = [summarize(path, args.warmup, args.period) for path in args.joint_logs]
    text = json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"Arm evaluation: {target.resolve()}")
    else:
        print(text)


if __name__ == "__main__":
    main()

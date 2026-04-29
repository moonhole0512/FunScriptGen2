import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np


def load_actions(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return sorted(data.get("actions", []), key=lambda action: int(action["at"]))


def interpolate(actions, times):
    if not actions:
        return np.zeros(len(times), dtype=np.float32) + 50.0

    action_times = np.array([int(action["at"]) for action in actions], dtype=np.float32)
    positions = np.array([int(action["pos"]) for action in actions], dtype=np.float32)
    return np.interp(times, action_times, positions, left=positions[0], right=positions[-1])


def sparse_extrema(actions, min_delta=12):
    if len(actions) < 3:
        return []

    extrema = []
    for idx in range(1, len(actions) - 1):
        previous_pos = int(actions[idx - 1]["pos"])
        current_pos = int(actions[idx]["pos"])
        next_pos = int(actions[idx + 1]["pos"])
        is_turn = (current_pos >= previous_pos and current_pos >= next_pos) or (
            current_pos <= previous_pos and current_pos <= next_pos
        )
        if is_turn and (abs(current_pos - previous_pos) >= min_delta or abs(current_pos - next_pos) >= min_delta):
            extrema.append(actions[idx])
    return extrema


def event_f1(auto_actions, human_actions, tolerance_ms=180):
    auto_extrema = sparse_extrema(auto_actions)
    human_extrema = sparse_extrema(human_actions)
    used = set()
    matched_offsets = []

    for auto_action in auto_extrema:
        best_idx = None
        best_distance = 10**9
        for idx, human_action in enumerate(human_extrema):
            if idx in used:
                continue
            distance = abs(int(auto_action["at"]) - int(human_action["at"]))
            if distance <= tolerance_ms and distance < best_distance:
                best_idx = idx
                best_distance = distance
        if best_idx is not None:
            used.add(best_idx)
            matched_offsets.append(int(auto_action["at"]) - int(human_extrema[best_idx]["at"]))

    true_positive = len(used)
    precision = true_positive / max(1, len(auto_extrema))
    recall = true_positive / max(1, len(human_extrema))
    f1 = 0.0 if precision + recall <= 1e-9 else (2 * precision * recall) / (precision + recall)
    return {
        "auto_extrema": len(auto_extrema),
        "human_extrema": len(human_extrema),
        "matched": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_offset_ms": median(matched_offsets) if matched_offsets else None,
    }


def score_series(auto_values, human_values):
    error = auto_values - human_values
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error * error)))
    position_similarity = max(0.0, 1.0 - (mae / 100.0))
    if float(np.std(auto_values)) <= 1e-6 or float(np.std(human_values)) <= 1e-6:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(auto_values, human_values)[0, 1])
    combined = (0.70 * position_similarity) + (0.30 * ((correlation + 1.0) / 2.0))
    return {
        "mae": mae,
        "rmse": rmse,
        "position_similarity": position_similarity,
        "correlation": correlation,
        "combined_similarity": combined,
    }


def compare_pair(auto_path, human_path, step_ms=20, max_shift_ms=1200):
    auto_actions = load_actions(auto_path)
    human_actions = load_actions(human_path)
    duration_ms = max(
        int(auto_actions[-1]["at"]) if auto_actions else 0,
        int(human_actions[-1]["at"]) if human_actions else 0,
    )
    times = np.arange(0, duration_ms + 1, step_ms, dtype=np.float32)
    human_values = interpolate(human_actions, times)

    base_values = interpolate(auto_actions, times)
    base = score_series(base_values, human_values)
    best = {"shift_ms": 0, "inverted": False, **base}

    shifts = range(-max_shift_ms, max_shift_ms + 1, step_ms)
    for inverted in (False, True):
        candidate_actions = [
            {"at": int(action["at"]), "pos": 100 - int(action["pos"]) if inverted else int(action["pos"])}
            for action in auto_actions
        ]
        for shift_ms in shifts:
            candidate_values = interpolate(candidate_actions, times + shift_ms)
            score = score_series(candidate_values, human_values)
            if score["combined_similarity"] > best["combined_similarity"]:
                best = {"shift_ms": shift_ms, "inverted": inverted, **score}

    return {
        "name": auto_path.stem,
        "auto_actions": len(auto_actions),
        "human_actions": len(human_actions),
        "duration_ms": duration_ms,
        "base": base,
        "best": best,
        "events": event_f1(auto_actions, human_actions),
    }


def format_percent(value):
    return f"{value * 100.0:.1f}%"


def print_result(result):
    base = result["base"]
    best = result["best"]
    events = result["events"]
    print(f"\n{result['name']}")
    print(f"  actions: auto={result['auto_actions']} human={result['human_actions']} duration={result['duration_ms']}ms")
    print(
        "  base: "
        f"combined={format_percent(base['combined_similarity'])} "
        f"pos={format_percent(base['position_similarity'])} "
        f"corr={base['correlation']:.3f} mae={base['mae']:.2f}"
    )
    print(
        "  best: "
        f"combined={format_percent(best['combined_similarity'])} "
        f"pos={format_percent(best['position_similarity'])} "
        f"corr={best['correlation']:.3f} mae={best['mae']:.2f} "
        f"shift={best['shift_ms']}ms inverted={best['inverted']}"
    )
    print(
        "  events: "
        f"f1={format_percent(events['f1'])} precision={format_percent(events['precision'])} "
        f"recall={format_percent(events['recall'])} matched={events['matched']}/"
        f"{events['human_extrema']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare generated funscripts with Video/HumanMade references.")
    parser.add_argument("--video-dir", default="Video")
    parser.add_argument("--human-dir", default="Video/HumanMade")
    parser.add_argument("--name", default=None, help="Optional base name without .funscript")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    human_dir = Path(args.human_dir)
    if args.name:
        pairs = [(video_dir / f"{args.name}.funscript", human_dir / f"{args.name}.funscript")]
    else:
        pairs = []
        for human_path in sorted(human_dir.glob("*.funscript")):
            auto_path = video_dir / human_path.name
            if auto_path.exists():
                pairs.append((auto_path, human_path))

    if not pairs:
        raise SystemExit("No matching funscript pairs found.")

    for auto_path, human_path in pairs:
        print_result(compare_pair(auto_path, human_path))


if __name__ == "__main__":
    main()

import argparse
import json
import os


WEIGHTS_SHORT = {
    "aesthetic": 0.10,
    "motion_amplitude": 0.10,
    "motion_smoothness": 0.10,
    "semantic": 0.35,
    "naturalness": 0.35,
}

WEIGHTS_LONG = {
    "aesthetic": 0.03,
    "motion_amplitude": 0.03,
    "motion_smoothness": 0.03,
    "semantic": 0.255,
    "naturalness": 0.255,
    "drifting_aesthetic": 0.099,
    "drifting_motion_smoothness": 0.099,
    "drifting_semantic": 0.099,
    "drifting_naturalness": 0.099,
}


METRIC_RANGES = {
    "aesthetic": {"min": 0, "max": 1},
    "motion_amplitude": {"min": 0, "max": 1},
    "motion_smoothness": {"min": 0, "max": 1},
    "naturalness": {"min": 0, "max": 1},
    "semantic": {"min": 0, "max": 1},
    "drifting_aesthetic": {"min": 0, "max": 1},
    "drifting_motion_smoothness": {"min": 0, "max": 1},
    "drifting_naturalness": {"min": 0, "max": 1},
    "drifting_semantic": {"min": 0, "max": 1},
}

METRIC_ALIASES = {
    "aesthetic_score": "aesthetic",
    "motion_amplitude_score": "motion_amplitude",
    "motion_smoothness_score": "motion_smoothness",
    "naturalness_score": "naturalness",
    "semantic_score": "semantic",
    "drift_aesthetic_score": "drifting_aesthetic",
    "drift_motion_smoothness_score": "drifting_motion_smoothness",
    "drift_naturalness_score": "drifting_naturalness",
    "drift_semantic_score": "drifting_semantic",
}

RATING_SCALE = 10

SCORING_RULES_SHORT = {
    "aesthetic": {
        "type": "higher_better",
        "thresholds": [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30],
    },
    "motion_amplitude": {
        "type": "higher_better",
        "thresholds": [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05],
    },
    "motion_smoothness": {
        "type": "higher_better",
        "thresholds": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91],
    },
    "semantic": {"type": "higher_better", "thresholds": [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22]},
    "naturalness": {"type": "higher_better", "thresholds": [0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.30, 0.25, 0.20]},
}

SCORING_RULES_LONG = {
    "aesthetic": {
        "type": "higher_better",
        "thresholds": [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.45, 0.40],
    },
    "motion_amplitude": {
        "type": "higher_better",
        "thresholds": [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05],
    },
    "motion_smoothness": {
        "type": "higher_better",
        "thresholds": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91],
    },
    "semantic": {"type": "higher_better", "thresholds": [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22]},
    "naturalness": {"type": "higher_better", "thresholds": [0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.30, 0.25, 0.20]},
    "drifting_aesthetic": {
        "type": "lower_better",
        "thresholds": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
    },
    "drifting_motion_smoothness": {
        "type": "lower_better",
        "thresholds": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
    },
    "drifting_semantic": {
        "type": "lower_better",
        "thresholds": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
    },
    "drifting_naturalness": {
        "type": "lower_better",
        "thresholds": [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22],
    },
}


def calculate_weighted_rating(summary_data, is_long):
    weights = WEIGHTS_LONG if is_long else WEIGHTS_SHORT
    total_rating = 0.0

    all_metrics = {**summary_data.get("non_drifting", {}), **summary_data.get("drifting", {})}

    print(f"\nCalculating Weighted Total Rating ({'Long' if is_long else 'Short'}):")
    for metric, weight in weights.items():
        if metric in all_metrics and all_metrics[metric]["rating"] is not None:
            contribution = all_metrics[metric]["rating"] * weight
            total_rating += contribution
            print(f"  - {metric}: {all_metrics[metric]['rating']} * {weight} = {contribution:.4f}")
        else:
            print(f"  ⚠ Warning: Metric '{metric}' missing for weighting!")

    return round(total_rating, 4)


def convert_to_rating(normalized_score, metric_key, is_long):
    if normalized_score is None:
        return None
    canonical_key = METRIC_ALIASES.get(metric_key, metric_key)
    rule = SCORING_RULES_LONG[canonical_key] if is_long else SCORING_RULES_SHORT[canonical_key]

    rule_type = rule["type"]
    thresholds = rule["thresholds"]

    if rule_type == "higher_better":
        for i, threshold in enumerate(thresholds):
            if normalized_score >= threshold:
                return RATING_SCALE - i
        return 1
    elif rule_type == "lower_better":
        for i, threshold in enumerate(thresholds):
            if normalized_score <= threshold:
                return RATING_SCALE - i
        return 1
    elif rule_type == "target_based":
        target = rule["target"]
        distance = abs(normalized_score - target)
        for i, threshold in enumerate(thresholds):
            if distance <= threshold:
                return RATING_SCALE - i
        return 1
    return None


def normalize_score(score, metric_key):
    if score is None or not isinstance(score, (int, float)):
        return None
    canonical_key = METRIC_ALIASES.get(metric_key, metric_key)
    if canonical_key not in METRIC_RANGES:
        return score
    min_val, max_val = METRIC_RANGES[canonical_key]["min"], METRIC_RANGES[canonical_key]["max"]
    normalized = (score - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def is_valid_metric(metric_key, is_long):
    canonical_key = METRIC_ALIASES.get(metric_key, metric_key)
    rules = SCORING_RULES_LONG if is_long else SCORING_RULES_SHORT
    return canonical_key in rules


def merge_results(input_dir, output_path=None, is_long=False):
    result_files = [f for f in os.listdir(input_dir) if f.endswith("_results.json")]
    if not result_files:
        return None, None

    merged = {
        "rating_scale": RATING_SCALE,
        "summary": {"non_drifting": {}, "drifting": {}},
        "per_video": {},
    }

    for result_file in sorted(result_files):
        metric_key = result_file.replace("_results.json", "")
        if not is_valid_metric(metric_key, is_long):
            continue

        try:
            with open(os.path.join(input_dir, result_file), "r") as f:
                data = json.load(f)

            score = data.get("average_score") or data.get("average_drift_score")
            norm_score = normalize_score(score, metric_key)
            rating = convert_to_rating(norm_score, metric_key, is_long)

            metric_summary = {
                "name": metric_key.replace("_", " ").title(),
                "raw_score": score,
                "normalized_score": norm_score,
                "rating": rating,
                "num_videos": data.get("num_videos", 0),
            }

            if metric_key.startswith("drifting_"):
                merged["summary"]["drifting"][metric_key] = metric_summary
            else:
                merged["summary"]["non_drifting"][metric_key] = metric_summary

        except Exception as e:
            print(f" Error loading {result_file}: {e}")

    total_weighted = calculate_weighted_rating(merged["summary"], is_long)
    merged["summary"]["total_weighted_rating"] = total_weighted
    print(f"\n>>> FINAL WEIGHTED RATING: {total_weighted}\n")

    if output_path is None:
        output_path = os.path.join(input_dir, "merged_results.json")
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    return merged, output_path


def main(args):
    merge_results(args.input_dir, args.output_path, args.is_long)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="playground/results/toy-video")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--is_long", action="store_true")
    args = parser.parse_args()
    main(args)

import argparse
import json
import os
from pathlib import Path


def collect_all_model_results(input_dir, score_type="normalized"):
    model_results = {}
    base_path = Path(input_dir)

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue

        merged_file = model_dir / "merged_results.json"
        if not merged_file.exists():
            print(f"⚠ Skipping {model_dir.name}: merged_results.json not found")
            continue

        try:
            with open(merged_file, "r") as f:
                data = json.load(f)

            model_name = model_dir.name
            scores = {}

            if score_type == "raw":
                score_field = "raw_score"
            elif score_type == "normalized":
                score_field = "normalized_score"
            elif score_type == "rating":
                score_field = "rating"
            else:
                score_field = "normalized_score"

            summary = data.get("summary", {})

            for metric_key, metric_info in summary.get("non_drifting", {}).items():
                scores[metric_key] = metric_info.get(score_field)

            for metric_key, metric_info in summary.get("drifting", {}).items():
                scores[metric_key] = metric_info.get(score_field)

            if "total_weighted_rating" in summary:
                scores["total_weighted_rating"] = summary["total_weighted_rating"]

            model_results[model_name] = scores
            print(f"✓ Loaded results for: {model_name}")

        except Exception as e:
            print(f"✗ Error loading {model_dir.name}/merged_results.json: {e}")

    return model_results


def create_final_merged_json(model_results, output_path, score_type="normalized", rating_scale=None):
    all_metrics = set()
    for scores in model_results.values():
        all_metrics.update(scores.keys())

    sorted_metrics = sorted([m for m in all_metrics if m != "total_weighted_rating"])
    if "total_weighted_rating" in all_metrics:
        sorted_metrics.append("total_weighted_rating")

    final_data = {
        "num_models": len(model_results),
        "score_type": score_type,
        "metrics": sorted_metrics,
        "models": model_results,
    }

    if score_type == "rating" and rating_scale is not None:
        final_data["rating_scale"] = rating_scale

    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2)

    return final_data


def get_rating_scale_from_results(input_dir):
    base_path = Path(input_dir)

    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue

        merged_file = model_dir / "merged_results.json"
        if merged_file.exists():
            try:
                with open(merged_file, "r") as f:
                    data = json.load(f)
                    rating_scale = data.get("rating_scale")
                    if rating_scale:
                        return rating_scale
            except Exception:
                continue

    return None


def main(args):
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return

    score_type = args.score_type

    rating_scale = None
    if score_type == "rating":
        rating_scale = get_rating_scale_from_results(input_dir)
        if rating_scale:
            score_type_str = f"RATING (1-{rating_scale})"
        else:
            score_type_str = "RATING"
    elif score_type == "normalized":
        score_type_str = "NORMALIZED"
    elif score_type == "raw":
        score_type_str = "RAW"
    else:
        score_type_str = score_type.upper()

    print(f"\n{'=' * 100}")
    print(f"ALL MODELS RESULTS MERGER (Using {score_type_str} Scores)")
    print(f"{'=' * 100}")
    print(f"Base Directory: {input_dir}\n")

    model_results = collect_all_model_results(input_dir, score_type)

    if not model_results:
        print("\n✗ No model results found!")
        return

    print(f"\n✓ Successfully loaded {len(model_results)} models\n")

    output_path = args.output_path or os.path.join(input_dir, "all_models_merged.json")
    create_final_merged_json(model_results, output_path, score_type, rating_scale)

    print(f"✓ All models merged results saved to: {output_path}\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all models' merged_results.json into a single comparison file")
    parser.add_argument("--input_dir", type=str, default="playground/results", help="Base directory")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--score_type", type=str, choices=["raw", "normalized", "rating"], default="rating")
    args = parser.parse_args()
    main(args)

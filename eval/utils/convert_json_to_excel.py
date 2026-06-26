import argparse
import json

import pandas as pd


CUSTOM_ORDER = [
    "total_weighted_rating",
    "aesthetic",
    "motion_amplitude",
    "motion_smoothness",
    "semantic",
    "naturalness",
    "drifting_aesthetic",
    "drifting_motion_smoothness",
    "drifting_semantic",
    "drifting_naturalness",
]

SELECTED_METRICS = [
    "total_weighted_rating",
    "aesthetic",
    "motion_amplitude",
    "motion_smoothness",
    "semantic",
    "naturalness",
]


def json_to_excel(json_path, excel_path=None, use_selected_metrics=False, show_raw_values=False, score_type=""):
    with open(json_path, "r") as f:
        data = json.load(f)

    models_data = data["models"]
    df = pd.DataFrame.from_dict(models_data, orient="index")

    df.reset_index(inplace=True)
    df.rename(columns={"index": "model_name"}, inplace=True)

    if use_selected_metrics:
        available_cols = ["model_name"] + [col for col in SELECTED_METRICS if col in df.columns]
        df = df[available_cols]
        print(f"Selected {len(available_cols) - 1} metrics from available metrics")

    valid_order = ["model_name"] + [col for col in CUSTOM_ORDER if col in df.columns]
    df = df[valid_order]
    print(f"Kept {len(valid_order) - 1} metrics as specified in CUSTOM_ORDER")

    if excel_path is None:
        excel_path = json_path.rsplit(".", 1)[0] + f"_{score_type}" + ".xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Models", index=False)

        metadata = pd.DataFrame(
            {
                "Property": ["timestamp", "num_models", "num_metrics", "filtered", "format"],
                "Value": [
                    data.get("timestamp", "N/A"),
                    data.get("num_models", len(models_data)),
                    len(df.columns) - 1,
                    "Yes" if use_selected_metrics else "No",
                    "Raw Values" if show_raw_values else "Percentage",
                ],
            }
        )
        metadata.to_excel(writer, sheet_name="Metadata", index=False)

        worksheet = writer.sheets["Models"]
        for idx, col in enumerate(df.columns):
            max_length = max(df[col].astype(str).apply(len).max(), len(col))
            if idx < 26:
                col_letter = chr(65 + idx)
            else:
                col_letter = chr(65 + idx // 26 - 1) + chr(65 + idx % 26)
            worksheet.column_dimensions[col_letter].width = min(max_length + 2, 50)

            if col != "model_name" and pd.api.types.is_numeric_dtype(df[col]):
                for row in range(2, len(df) + 2):  # Start from row 2 (after header)
                    cell = worksheet[f"{col_letter}{row}"]
                    if cell.value is not None:
                        if col == "total_weighted_rating":
                            cell.number_format = "0.00"
                        elif show_raw_values:
                            cell.number_format = "0"
                        else:
                            cell.value = cell.value * 100
                            cell.number_format = '0.00"%"'

    print(f"Conversion successful! Output file: {excel_path}")
    print(f"Processed {len(df)} models with {len(df.columns) - 1} metrics")
    print(f"Format: {'Raw values' if show_raw_values else 'Percentage'}")

    return excel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_file", type=str, required=True, help="Input JSON file path")
    parser.add_argument(
        "--excel_file",
        type=str,
        required=True,
        help="Output Excel file path (optional, defaults to input filename.xlsx)",
    )
    parser.add_argument("--filter", action="store_true", help="Use only metrics defined in SELECTED_METRICS list")
    parser.add_argument(
        "--score_type",
        type=str,
        choices=["raw", "normalized", "rating"],
        default="rating",
        help="Type of scores to use: 'raw', 'normalized', or 'rating'",
    )

    args = parser.parse_args()

    if args.score_type == "rating":
        raw_value = True
    else:
        raw_value = False

    try:
        json_to_excel(
            args.json_file,
            args.excel_file,
            use_selected_metrics=args.filter,
            show_raw_values=raw_value,
            score_type=args.score_type,
        )
    except FileNotFoundError:
        print(f"Error: File not found {args.json_file}")
    except json.JSONDecodeError:
        print(f"Error: {args.json_file} is not a valid JSON file")
    except Exception as e:
        print(f"Error: {e}")

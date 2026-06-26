import yaml


def compare_yaml(file1_path, file2_path):
    with open(file1_path, "r") as f1:
        yaml1 = yaml.safe_load(f1)

    with open(file2_path, "r") as f2:
        yaml2 = yaml.safe_load(f2)

    missing_keys = []
    different_values = []

    compare_dict(yaml1, yaml2, "", missing_keys, different_values)

    print("=" * 60)
    print("Missing Keys")
    print("=" * 60)
    if missing_keys:
        for diff in missing_keys:
            print(diff)
    else:
        print("None")

    print("\n" + "=" * 60)
    print("Different Values")
    print("=" * 60)
    if different_values:
        for diff in different_values:
            print(diff)
    else:
        print("None")

    print("\n" + "=" * 60)
    print(f"Total: {len(missing_keys)} missing keys, {len(different_values)} different values")
    print("=" * 60)


def compare_dict(dict1, dict2, path, missing_keys, different_values):
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        current_path = f"{path}.{key}" if path else key

        if key not in dict2:
            missing_keys.append(f"[{current_path}] Only in file1: {dict1[key]}")
        elif key not in dict1:
            missing_keys.append(f"[{current_path}] Only in file2: {dict2[key]}")
        else:
            val1, val2 = dict1[key], dict2[key]

            if isinstance(val1, dict) and isinstance(val2, dict):
                compare_dict(val1, val2, current_path, missing_keys, different_values)
            elif isinstance(val1, list) and isinstance(val2, list):
                if val1 != val2:
                    different_values.append(f"[{current_path}]\n  File1: {val1}\n  File2: {val2}")
            elif val1 != val2:
                different_values.append(f"[{current_path}]\n  File1: {val1}\n  File2: {val2}")


if __name__ == "__main__":
    compare_yaml(
        "configs/stage_1_init.yaml",
        "configs/stage_1_post.yaml",
    )

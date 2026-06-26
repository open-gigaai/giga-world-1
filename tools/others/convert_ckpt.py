import json
import os

from safetensors import safe_open
from safetensors.torch import save_file


# RENAME_RULES = [
#     ("clean_patch_embedding.proj_4x", "multi_term_memory_patch.patch_long"),
#     ("clean_patch_embedding.proj_2x", "multi_term_memory_patch.patch_mid"),
#     ("clean_patch_embedding.proj", "multi_term_memory_patch.patch_short"),
# ]

# RENAME_RULES = [
#     ("multi_term_memory_patch.proj_4x", "multi_term_memory_patch.patch_long"),
#     ("multi_term_memory_patch.proj_2x", "multi_term_memory_patch.patch_mid"),
#     ("multi_term_memory_patch.proj",    "multi_term_memory_patch.patch_short"),
# ]

# RENAME_RULES = [
#     ("multi_term_memory_patch.patch_long", "patch_long"),
#     ("multi_term_memory_patch.patch_mid", "patch_mid"),
#     ("multi_term_memory_patch.patch_short", "patch_short"),
# ]

# RENAME_RULES = [
#     ("multi_term_memory_patch.patch_long", "patch_long"),
#     ("multi_term_memory_patch.patch_mid", "patch_mid"),
#     ("multi_term_memory_patch.patch_short", "patch_short"),
# ]

RENAME_RULES = [
    ("scale_shift_table", "norm_out.scale_shift_table"),
]


BASE_DIRS = [
    "Helios-Base/transformer",
    "Helios-Base/transformer_init",
    "Helios-Base-init/transformer",
    "Helios-Mid/transformer",
    "Helios-Mid/transformer_init",
    "Helios-Mid-init/transformer",
    "Helios-Distilled/transformer",
    "Helios-Distilled/transformer_ode",
    "Helios-Distilled-ODE/transformer",
]

for BASE_DIR in BASE_DIRS:
    index_path = os.path.join(BASE_DIR, "diffusion_pytorch_model.safetensors.index.json")

    def apply_rename(key: str) -> str:
        for old_prefix, new_prefix in RENAME_RULES:
            if key.startswith(old_prefix):
                return new_prefix + key[len(old_prefix) :]
        return key

    print(f"[1/3] Reading {index_path} ...")
    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    old_keys = [k for k in weight_map if apply_rename(k) != k]
    print(f"      Found {len(old_keys)} keys to rename")

    for old_key in old_keys:
        new_key = apply_rename(old_key)
        weight_map[new_key] = weight_map.pop(old_key)
        print(f"        {old_key}  ->  {new_key}")

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print("      Saved updated index.json")

    with open(index_path, "r") as f:
        updated_index = json.load(f)

    affected_shards = set()
    for new_key, shard in updated_index["weight_map"].items():
        if any(new_key.startswith(new_prefix) for _, new_prefix in RENAME_RULES):
            affected_shards.add(shard)

    print(f"\n[2/3] Affected shard files: {sorted(affected_shards)}")

    for shard_filename in sorted(affected_shards):
        shard_path = os.path.join(BASE_DIR, shard_filename)
        print(f"\n      Processing {shard_filename} ...")

        tensors = {}
        metadata = None

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for key in f.keys():
                new_key = apply_rename(key)
                tensors[new_key] = f.get_tensor(key)
                if key != new_key:
                    print(f"        {key}  ->  {new_key}")

        save_file(tensors, shard_path, metadata=metadata)
        print(f"      Overwritten {shard_path}")

    print("\n[3/3] Done!")

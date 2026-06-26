import os
import re
import json
import glob
from collections import defaultdict
from tqdm import tqdm

# FOR WAN 21 CTRL
# JSONL_PATH = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view"
# PT_ROOT = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view/latents_short_giga_control"
# SAVE_DIR = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view/task_pt_json"

# FOR WAN 22 5B CTRL
# JSONL_PATH = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view"
# PT_ROOT = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view_wan22_5b/latents_short_giga_control"
# SAVE_DIR = "/shared_disk/users/zhanqian.wu/data/train_data/helios_data/giga_ctrl_3view_wan22_5b/task_pt_json"

JSONL_PATH = "/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl/"
PT_ROOT = "/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl/latents_short_giga_control"
SAVE_DIR = "/shared_disk/users/zhanqian.wu/data/CC_Train_WM/helios_giga_ctrl/task_pt_json"

# def iter_jsonl_files(path):
#     if os.path.isfile(path):
#         return [path]

#     jsonl_files = []
#     pbar = tqdm(desc="scanning jsonl files", unit="files", dynamic_ncols=True)

#     for dirpath, _, filenames in os.walk(path):
#         print(dirpath)
#         for fn in filenames:
#             if fn.endswith(".jsonl"):
#                 jsonl_files.append(os.path.join(dirpath, fn))
#                 pbar.update(1)

#     pbar.close()
#     return sorted(jsonl_files)
def iter_jsonl_files(path):
    # 如果传入本身就是单个jsonl文件，直接返回
    if os.path.isfile(path) and path.endswith(".jsonl"):
        return [path]

    jsonl_files = []
    pbar = tqdm(desc="scanning jsonl files", unit="files", unit_scale=True, dynamic_ncols=True)

    # os.listdir 只遍历当前目录，不递归子文件夹
    for fname in os.listdir(path):
        full_path = os.path.join(path, fname)
        # 只匹配当前目录下的 .jsonl 文件，跳过所有子文件夹
        if os.path.isfile(full_path) and fname.endswith(".jsonl"):
            jsonl_files.append(full_path)
            pbar.update(1)

    pbar.close()
    return sorted(jsonl_files)

def extract_task_from_source(item):
    source_info = item.get("source_info", {})

    for p in source_info.values():
        if not p:
            continue

        parts = str(p).replace("\\", "/").split("/")

        if "ctrl" in parts:
            idx = parts.index("ctrl")
            if idx + 1 < len(parts):
                return parts[idx + 1]

    p = item.get("path", "")
    if p:
        return str(p).replace("\\", "/").split("/")[0]

    return None

def extract_uuid_from_path(path):
    """
    支持:
    task1/episode_000001_ac2a1748fc_s000000_e000129.mp4
    episode_000000_0f2cf1e845_s000903_e001032_0-129_121_480_1920.pt
    """
    name = os.path.basename(str(path))

    m = re.search(
        r"episode_\d+_([0-9a-fA-F]+)_s\d+_e\d+",
        name,
    )

    if m:
        return m.group(1)

    return None

def count_jsonl_lines(jsonl_files):
    total = 0

    for path in tqdm(jsonl_files, desc="count jsonl lines", dynamic_ncols=True):
        with open(path, "r") as f:
            for _ in f:
                total += 1

    return total

def build_task_uuid_map(jsonl_files):
    """
    第一步:
    从 jsonl 里建立:
        task -> uuid set
    """

    task_to_uuids = defaultdict(set)
    uuid_to_tasks = defaultdict(set)

    bad_json = 0
    no_task = 0
    no_uuid = 0
    total_valid = 0

    total_lines = count_jsonl_lines(jsonl_files)

    pbar = tqdm(
        total=total_lines,
        desc="build task->uuid",
        dynamic_ncols=True,
    )

    for jsonl_path in jsonl_files:
        with open(jsonl_path, "r") as f:
            for line in f:
                pbar.update(1)

                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except Exception:
                    bad_json += 1
                    continue

                task = extract_task_from_source(item)
                uuid = extract_uuid_from_path(item.get("path", ""))

                if not task:
                    no_task += 1
                    continue

                if not uuid:
                    no_uuid += 1
                    continue

                task_to_uuids[task].add(uuid)
                uuid_to_tasks[uuid].add(task)
                total_valid += 1

    pbar.close()

    print("")
    print("📊 jsonl scan summary")
    print(f"✅ valid json items: {total_valid}")
    print(f"✅ task count:       {len(task_to_uuids)}")
    print(f"✅ uuid count:       {len(uuid_to_tasks)}")
    print(f"⚠️ bad json:         {bad_json}")
    print(f"⚠️ no task:          {no_task}")
    print(f"⚠️ no uuid:          {no_uuid}")

    for task in sorted(task_to_uuids.keys()):
        print(f"  - {task}: uuid={len(task_to_uuids[task])}")

    return task_to_uuids, uuid_to_tasks

def scan_pt_files(pt_root):
    pt_files = []
    stack = [pt_root]
    pbar = tqdm(desc="scanning entries", unit="entries", unit_scale=True, dynamic_ncols=True)

    while stack:
        dirpath = stack.pop()
        try:
            with os.scandir(dirpath) as entries:
                for entry in entries:
                    pbar.update(1)
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".pt"):
                            pt_files.append(entry.path)
                            pbar.set_postfix(pt=len(pt_files), refresh=False)
                    except OSError:
                        continue
        except OSError as e:
            tqdm.write(f"⚠️ skip dir: {dirpath} ({e})")
            continue

    pbar.close()
    pt_files.sort()

    print("")
    print(f"✅ total pt files: {len(pt_files)}")

    return pt_files

def build_uuid_pt_map(pt_files):
    """
    第二步:
    扫描 pt，建立:
        uuid -> pt list
    """

    uuid_to_pts = defaultdict(list)

    no_uuid = 0

    for pt in tqdm(pt_files, desc="build uuid->pt", dynamic_ncols=True):
        uuid = extract_uuid_from_path(pt)

        if not uuid:
            no_uuid += 1
            continue

        uuid_to_pts[uuid].append(pt)

    print("")
    print("📊 pt scan summary")
    print(f"✅ pt uuid count: {len(uuid_to_pts)}")
    print(f"⚠️ pt no uuid:    {no_uuid}")

    return uuid_to_pts

def export_task_jsons(task_to_uuids, uuid_to_pts):
    """
    第三步:
    每个 task 保存一个 json
    """

    os.makedirs(SAVE_DIR, exist_ok=True)

    total_export_pt = 0
    total_missing_uuid = 0

    for task, uuid_set in tqdm(
        sorted(task_to_uuids.items()),
        desc="export task json",
        dynamic_ncols=True,
    ):
        matched_pts = []
        missing_uuids = []

        for uuid in tqdm(sorted(uuid_set), desc=f"  match {task}", leave=False, dynamic_ncols=True):
            pts = uuid_to_pts.get(uuid, [])

            if not pts:
                missing_uuids.append(uuid)
                continue

            matched_pts.extend(pts)

        matched_pts = sorted(set(matched_pts))

        out = {
            "task": task,
            "num_uuid": len(uuid_set),
            "uuids": sorted(uuid_set),
            "num_matched_uuid": len(uuid_set) - len(missing_uuids),
            "num_missing_uuid": len(missing_uuids),
            "missing_uuids": missing_uuids,
            "num_pt": len(matched_pts),
            "pt_paths": matched_pts,
        }

        save_path = os.path.join(SAVE_DIR, f"{task}.json")

        with open(save_path, "w") as f:
            json.dump(
                out,
                f,
                indent=4,
                ensure_ascii=False,
            )

        total_export_pt += len(matched_pts)
        total_missing_uuid += len(missing_uuids)

        tqdm.write(
            f"✅ {task}: uuid={len(uuid_set)}, "
            f"matched_uuid={len(uuid_set) - len(missing_uuids)}, "
            f"missing_uuid={len(missing_uuids)}, "
            f"pt={len(matched_pts)}"
        )

    print("")
    print("🎉 done")
    print(f"✅ save dir:           {SAVE_DIR}")
    print(f"✅ total export pt:    {total_export_pt}")
    print(f"⚠️ total missing uuid: {total_missing_uuid}")

def main():
    print("📦 finding jsonl files...")
    jsonl_files = iter_jsonl_files(JSONL_PATH)
    print(f"✅ found jsonl files: {len(jsonl_files)}")

    task_to_uuids, uuid_to_tasks = build_task_uuid_map(jsonl_files)

    pt_files = scan_pt_files(PT_ROOT)
    uuid_to_pts = build_uuid_pt_map(pt_files)

    export_task_jsons(
        task_to_uuids=task_to_uuids,
        uuid_to_pts=uuid_to_pts,
    )


if __name__ == "__main__":
    main()
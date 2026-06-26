import os


os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["DIFFUSERS_ENABLE_HUB_KERNELS"] = "yes"

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from helios.utils.utils_base import encode_prompt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel


def setup_distributed_env():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def check_file_exists(args):
    basename, idx, line, output_folder = args
    uttid = f"{basename}_{idx:05d}"
    output_path = os.path.join(output_folder, f"{uttid}.pt")
    if os.path.exists(output_path):
        return None, None
    return line.strip(), uttid


def prepare_dataset_on_rank0(txt_file, output_folder, rank):
    while True:
        try:
            if rank == 0:
                basename = Path(txt_file).stem
                output_dir = Path(output_folder)

                existing_files = set()
                if output_dir.exists():
                    existing_files = {f.name for f in output_dir.iterdir() if f.is_file()}

                prompts = []
                uttids = []

                with open(txt_file, "r") as f:
                    for idx, line in enumerate(f):
                        if not line.strip():
                            continue

                        uttid = f"{basename}_{idx:05d}"
                        filename = f"{uttid}.pt"

                        if filename not in existing_files:
                            prompts.append(line.strip())
                            uttids.append(uttid)

                data_to_broadcast = [prompts, uttids]
            else:
                data_to_broadcast = [None, None]

            dist.broadcast_object_list(data_to_broadcast, src=0)
            break
        except Exception:
            continue

    return data_to_broadcast[0], data_to_broadcast[1]


class PromptDataset(Dataset):
    def __init__(self, prompts, uttids):
        self.prompts = prompts
        self.uttids = uttids

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "uttid": self.uttids[idx]}


def save_single_file(uttid, output_path, prompt_raw, prompt_embed):
    temp_to_save = {
        "prompt_raw": prompt_raw,
        "prompt_embed": prompt_embed,
    }

    try:
        torch.save(temp_to_save, output_path, pickle_protocol=4)
        return f"✓ Saved: {output_path}"
    except Exception as e:
        return f"✗ Failed to save {uttid}: {str(e)}"


def main():
    save_executor = ThreadPoolExecutor(max_workers=8)
    save_futures = []

    args = parse_args()

    # =============== Environment ===============
    batch_size = 16
    dataloader_num_workers = 8
    feature_folders = [
        "example/vidprom_first_1k.txt",
    ]
    output_folders = [
        "example/toy_data/text-embedding/vidprom_filtered_extended",
    ]

    if args.weight_dtype == "fp32":
        args.weight_dtype = torch.float32
    elif args.weight_dtype == "fp16":
        args.weight_dtype = torch.float16
    else:
        args.weight_dtype = torch.bfloat16

    setup_distributed_env()

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()

    accelerator = Accelerator()

    # =============== Prepare Model ===============
    weight_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        subfolder="tokenizer",
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.base_model_path,
        subfolder="text_encoder",
        dtype=weight_dtype,
    )

    text_encoder.eval()
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.to(device)

    for feature_folder, output_folder in zip(feature_folders, output_folders):
        print(f"Process {feature_folder} !")

        os.makedirs(output_folder, exist_ok=True)
        prompts, uttids = prepare_dataset_on_rank0(feature_folder, output_folder, rank)
        dataset = PromptDataset(prompts, uttids)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_num_workers,
            prefetch_factor=2 if dataloader_num_workers > 0 else None,
            pin_memory=True,
            drop_last=False,
        )
        dataloader = accelerator.prepare(dataloader)
        print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")
        print(f"Process index: {accelerator.process_index}, World size: {accelerator.num_processes}")

        if len(dataloader) == 0:
            continue

        # =============== Main Loop ===============
        if rank == 0:
            pbar = tqdm(total=len(dataloader), desc="Processing")

        for i, batch in enumerate(dataloader):
            batch_size = len(batch["uttid"])
            uttids = batch["uttid"]
            prompts_raw = batch["prompt"]

            files_to_process = []
            indices_to_process = []

            for idx, uttid in enumerate(uttids):
                output_path = os.path.join(output_folder, f"{uttid}.pt")
                if os.path.exists(output_path):
                    if rank == 0:
                        print(f"Skipping existing file: {output_path}")
                else:
                    files_to_process.append((uttid, output_path))
                    indices_to_process.append(idx)

            if len(files_to_process) == 0:
                if rank == 0:
                    pbar.update(1)
                continue

            prompts_to_encode = [prompts_raw[idx] for idx in indices_to_process]

            with torch.no_grad():
                prompt_embeds, _ = encode_prompt(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=prompts_to_encode,
                    device=device,
                )

            for idx, (uttid, output_path) in enumerate(files_to_process):
                prompt_embed_cpu = prompt_embeds[idx].cpu().clone()

                future = save_executor.submit(
                    save_single_file, uttid, output_path, prompts_to_encode[idx], prompt_embed_cpu
                )
                save_futures.append(future)

            if len(save_futures) > 100:
                completed_futures = [f for f in save_futures if f.done()]

                if rank == 0:
                    for future in completed_futures:
                        try:
                            result = future.result()
                            print(result)
                        except Exception as e:
                            print(f"Save task error: {e}")

                save_futures = [f for f in save_futures if not f.done()]

            if rank == 0:
                pbar.update(1)

        if rank == 0:
            pbar.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video with model")

    # === Model paths ===
    parser.add_argument("--base_model_path", type=str, default="BestWishYsh/Helios-Base")

    # === Generation parameters ===
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for model weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")

    # === Prompts ===
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()

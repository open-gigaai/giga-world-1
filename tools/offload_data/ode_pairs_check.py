import os
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video


def stat_tensor(name, x):
    if x is None:
        print(f"{name}: None")
        return True

    ok = True
    if not torch.is_tensor(x):
        print(f"{name}: not tensor, type={type(x)}")
        return False

    finite = torch.isfinite(x).all().item()
    if not finite:
        ok = False

    print(
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
        f"mean={x.float().mean().item():.6f} "
        f"std={x.float().std().item():.6f} "
        f"min={x.float().min().item():.6f} "
        f"max={x.float().max().item():.6f} "
        f"finite={finite}"
    )
    return ok


def normalize_latent_for_wan_vae(vae, latents):
    """
    输入 latent: [B,C,T,H,W] 或 [C,T,H,W]
    你的保存代码里 gt_vae_latent / ode_latents 都是原始 VAE latent，
    decode 前需要:
        latent / latents_std + latents_mean
    """
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)

    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    latents = latents.to(device=device, dtype=dtype)

    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=device,
        dtype=dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)

    latents_std = 1.0 / torch.tensor(
        vae.config.latents_std,
        device=device,
        dtype=dtype,
    ).view(1, vae.config.z_dim, 1, 1, 1)

    return latents / latents_std + latents_mean


@torch.no_grad()
def decode_latent_to_mp4(vae, video_processor, latents, save_path, fps=20):
    latents = normalize_latent_for_wan_vae(vae, latents)
    video = vae.decode(latents, return_dict=False)[0]
    video = video_processor.postprocess_video(video, output_type="pil")
    export_to_video(video[0], save_path, fps=fps)
    print(f"saved video: {save_path}")


def check_ode_structure(item):
    ok = True
    ode = item.get("ode_latents", None)

    if ode is None:
        print("❌ missing ode_latents")
        return False

    if not isinstance(ode, list):
        print(f"❌ ode_latents should be list, got {type(ode)}")
        return False

    print(f"ode sections: {len(ode)}")

    for sec_i, section in enumerate(ode):
        if not isinstance(section, list):
            print(f"❌ ode_latents[{sec_i}] should be list")
            ok = False
            continue

        print(f"  section {sec_i}: stages={len(section)}")

        for stage_i, ode_item in enumerate(section):
            if not isinstance(ode_item, dict):
                print(f"❌ ode[{sec_i}][{stage_i}] should be dict")
                ok = False
                continue

            print(f"    stage {stage_i}: keys={list(ode_item.keys())}")

            latents = ode_item.get("latents", None)
            timesteps = ode_item.get("timesteps", None)

            ok &= stat_tensor(f"ode[{sec_i}][{stage_i}].latents", latents)
            ok &= stat_tensor(f"ode[{sec_i}][{stage_i}].timesteps", timesteps)

            if latents is not None and timesteps is not None:
                # 你保存时 latents = selected timesteps + 最后 x0
                if latents.shape[0] != timesteps.shape[0] + 1:
                    print(
                        f"❌ length mismatch: latents[0]={latents.shape[0]}, "
                        f"timesteps[0]={timesteps.shape[0]}, expected latents = timesteps + 1"
                    )
                    ok = False

            for k in ["noise_pred", "control_latents", "model_input"]:
                if k in ode_item:
                    ok &= stat_tensor(f"ode[{sec_i}][{stage_i}].{k}", ode_item[k])

    return ok


def pick_latent_for_vis(item, sec_i=-1, stage_i=-1, latent_i=-1):
    ode_item = item["ode_latents"][sec_i][stage_i]
    latents = ode_item["latents"]

    x = latents[latent_i]

    # 常见保存形状: [N, 1, 16, T, H, W]
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, default="/shared_disk/users/zhanqian.wu/data/train_data/debug/helios_data_stage3_FlowMatchEulerDiscreteScheduler")
    parser.add_argument("--base_model_path", type=str, default="/mnt/pfs/users/zhanqian.wu/ckpt/stage-1/stage1_final")
    parser.add_argument("--out_dir", type=str, default="/shared_disk/users/zhanqian.wu/data/train_data/debug/helios_data_stage3/")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--decode", action="store_false", default=True)
    parser.add_argument("--max_files", type=int, default=999999)
    args = parser.parse_args()

    pt_path = Path(args.pt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pt_path.is_dir():
        pt_files = sorted(pt_path.glob("*.pt"))[: args.max_files]
    else:
        pt_files = [pt_path]

    vae = None
    video_processor = None

    if args.decode:
        vae = AutoencoderKLWan.from_pretrained(
            args.base_model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(args.device)
        vae.eval()
        vae.requires_grad_(False)
        vae.enable_tiling()

        video_processor = VideoProcessor(
            vae_scale_factor=vae.spatial_compression_ratio
        )

    all_ok = True

    for path in tqdm(pt_files):
        print("\n" + "=" * 100)
        print(f"checking: {path}")

        try:
            item = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"❌ torch.load failed: {e}")
            all_ok = False
            continue

        print(f"top keys: {list(item.keys())}")
        print(f"prompt_raw: {item.get('prompt_raw', None)}")
        print(f"source_pt: {item.get('source_pt', None)}")
        print(f"latent_window_size: {item.get('latent_window_size', None)}")

        required_keys = [
            "prompt_raw",
            "prompt_embed",
            "gt_vae_latent",
            "control_latent",
            "ode_latents",
            "source_pt",
        ]

        for k in required_keys:
            if k not in item:
                print(f"❌ missing key: {k}")
                all_ok = False

        all_ok &= stat_tensor("prompt_embed", item.get("prompt_embed", None))
        all_ok &= stat_tensor("gt_vae_latent", item.get("gt_vae_latent", None))
        all_ok &= stat_tensor("control_latent", item.get("control_latent", None))

        all_ok &= check_ode_structure(item)

        if args.decode:
            stem = path.stem[:40]
            sample_dir = out_dir / stem
            sample_dir.mkdir(parents=True, exist_ok=True)

            # 1. GT
            if "gt_vae_latent" in item:
                gt = item["gt_vae_latent"]
                if gt.ndim == 4:
                    gt = gt.unsqueeze(0)
                decode_latent_to_mp4(
                    vae,
                    video_processor,
                    gt,
                    str(sample_dir / "gt_vae_latent.mp4"),
                    fps=args.fps,
                )

            # 2. ODE 最终 x0
            try:
                ode_x0 = pick_latent_for_vis(item, sec_i=-1, stage_i=-1, latent_i=-1)
                decode_latent_to_mp4(
                    vae,
                    video_processor,
                    ode_x0,
                    str(sample_dir / "ode_last_section_last_stage_x0.mp4"),
                    fps=args.fps,
                )
            except Exception as e:
                print(f"❌ decode ode x0 failed: {e}")
                all_ok = False

            # 3. 第一段每个 stage 的最后 x0
            try:
                for sec_i, section in enumerate(item["ode_latents"]):
                    sec_dir = sample_dir / f"section_{sec_i}"
                    sec_dir.mkdir(exist_ok=True)

                    for stage_i, ode_item in enumerate(section):

                        stage_dir = sec_dir / f"stage_{stage_i}"
                        stage_dir.mkdir(exist_ok=True)

                        latents = ode_item["latents"]
                        timesteps = ode_item["timesteps"]

                        print(
                            f"section={sec_i} stage={stage_i} "
                            f"num_saved={latents.shape[0]}"
                        )

                        for latent_i in range(latents.shape[0]):

                            x = latents[latent_i]

                            if x.ndim == 5 and x.shape[0] == 1:
                                x = x[0]

                            if latent_i < len(timesteps):
                                t = float(timesteps[latent_i])
                                save_name = f"t_{t:.1f}.mp4"
                            else:
                                save_name = "x0.mp4"

                            decode_latent_to_mp4(
                                vae,
                                video_processor,
                                x,
                                str(stage_dir / save_name),
                                fps=args.fps,
                            )
            except Exception as e:
                print(f"⚠️ decode section0 stages failed: {e}")

    print("\n" + "=" * 100)
    print("✅ ALL OK" if all_ok else "❌ FOUND ERROR")


if __name__ == "__main__":
    main()
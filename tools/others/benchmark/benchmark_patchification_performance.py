import os


os.environ["DIFFUSERS_ENABLE_HUB_KERNELS"] = "yes"

import json
import time
from datetime import datetime

import torch

from diffusers import WanTransformer3DModel


# 加载transformer
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
transformer.enable_gradient_checkpointing()
transformer.set_attention_backend("_flash_3_hub")
transformer.to("cuda")

noise_per_token = 960
noise_total_token = noise_per_token * 9

his_tokens = [960, 1920, 3840, 5760, 7680, 9600, 11520, 13440, 15360, 17280]
his_tokens_naive = [960, 1920, 2160, 2190, 2220, 2250, 2280, 2310, 2340, 2370]

benchmark_results = {
    "timestamp": datetime.now().isoformat(),
    "noise_total_token": noise_total_token,
    "experiments": [],
}


def create_dummy_inputs(transformer, num_frames, height=384, width=640, requires_grad=False):
    """创建transformer的dummy输入"""
    batch_size = 1
    device = transformer.device
    dtype = transformer.dtype

    # hidden_states: [B, C, F, H, W]
    in_channels = transformer.config.in_channels
    latent_h = height // 8
    latent_w = width // 8
    latent_f = num_frames

    hidden_states = torch.randn(
        batch_size, in_channels, latent_f, latent_h, latent_w, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # timestep
    timestep = torch.tensor([999], device=device, dtype=torch.long)
    timestep = timestep.expand(batch_size)

    # encoder_hidden_states
    seq_len = 512
    hidden_dim = 4096
    encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    return hidden_states, timestep, encoder_hidden_states


def measure_inference_speed(transformer, hidden_states, timestep, encoder_hidden_states, num_runs=10):
    """测量推理速度(单步)"""
    try:
        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=True,
                )
        torch.cuda.synchronize()

        # 正式测速
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                _ = transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=True,
                )

            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

        return {
            "avg_time_s": round(sum(times) / len(times), 4),
            "min_time_s": round(min(times), 4),
            "max_time_s": round(max(times), 4),
            "std_time_s": round(torch.std(torch.tensor(times)).item(), 4),
            "status": "success",
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return {"status": "OOM", "error": str(e)}
        else:
            raise


def measure_inference_memory(transformer, hidden_states, timestep, encoder_hidden_states):
    """测量推理显存"""
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3

        # Forward (推理模式)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True,
                attention_kwargs=None,
            )
        torch.cuda.synchronize()

        inference_peak = torch.cuda.max_memory_allocated() / 1024**3
        inference_mem_diff = inference_peak - mem_before

        return {
            "mem_before_gb": round(mem_before, 3),
            "inference_peak_gb": round(inference_peak, 3),
            "inference_mem_diff_gb": round(inference_mem_diff, 3),
            "status": "success",
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return {"status": "OOM", "error": str(e)}
        else:
            raise


def measure_training_memory(transformer, hidden_states, timestep, encoder_hidden_states):
    """测量训练显存(包含backward)"""
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3

        # Forward + Backward (训练模式)
        torch.cuda.reset_peak_memory_stats()

        # Forward
        output = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
            attention_kwargs=None,
        )

        # 创建一个简单的loss并backward
        loss = output.sample.sum()
        loss.backward()

        torch.cuda.synchronize()

        training_peak = torch.cuda.max_memory_allocated() / 1024**3
        training_mem_diff = training_peak - mem_before

        # 清理梯度
        transformer.zero_grad(set_to_none=True)

        return {
            "mem_before_gb": round(mem_before, 3),
            "training_peak_gb": round(training_peak, 3),
            "training_mem_diff_gb": round(training_mem_diff, 3),
            "status": "success",
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            transformer.zero_grad(set_to_none=True)
            return {"status": "OOM", "error": str(e)}
        else:
            raise


def warmup(transformer, num_runs=3):
    """预热"""
    print("🔥 Warming up...")
    for i in range(num_runs):
        hidden_states, timestep, encoder_hidden_states = create_dummy_inputs(transformer, num_frames=5)
        with torch.no_grad():
            _ = transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True,
            )
        print(f"  Warmup {i + 1}/{num_runs} done")
    torch.cuda.empty_cache()
    print("✅ Warmup completed\n")


def run_experiment(his_tokens_list, experiment_name):
    """运行完整实验"""
    results = []

    for his_token in his_tokens_list:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        total_token = his_token + noise_total_token
        num_frames = round((total_token / noise_per_token - 1) * 4 + 1)

        print(f"\n{'=' * 60}")
        print(f"{experiment_name} | tokens: {his_token} | frames: {int(num_frames)}")
        print(f"{'=' * 60}")

        result = {
            "his_token": his_token,
            "total_token": total_token,
            "num_frames": int(num_frames),
        }

        # 1. 测推理速度 (不需要梯度)
        print("📊 Measuring inference speed...")
        try:
            hidden_states, timestep, encoder_hidden_states = create_dummy_inputs(
                transformer, num_frames, requires_grad=False
            )
            speed_stats = measure_inference_speed(transformer, hidden_states, timestep, encoder_hidden_states)

            if speed_stats["status"] == "OOM":
                print("   ❌ OOM - Skipping remaining tests for this config")
                result.update({"speed_status": "OOM", "inference_status": "SKIPPED", "training_status": "SKIPPED"})
                results.append(result)
                del hidden_states, timestep, encoder_hidden_states
                torch.cuda.empty_cache()
                continue
            else:
                print(
                    f"   Avg: {speed_stats['avg_time_s']:.4f}s | "
                    f"Min: {speed_stats['min_time_s']:.4f}s | "
                    f"Max: {speed_stats['max_time_s']:.4f}s"
                )
                result.update(speed_stats)

            del hidden_states, timestep, encoder_hidden_states
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ Error: {e}")
            result["speed_status"] = "ERROR"
            torch.cuda.empty_cache()

        # 2. 测推理显存 (不需要梯度)
        print("💾 Measuring inference memory...")
        try:
            hidden_states, timestep, encoder_hidden_states = create_dummy_inputs(
                transformer, num_frames, requires_grad=False
            )
            inference_mem_stats = measure_inference_memory(transformer, hidden_states, timestep, encoder_hidden_states)

            if inference_mem_stats["status"] == "OOM":
                print("   ❌ OOM - Skipping training test")
                result.update(inference_mem_stats)
                result["training_status"] = "SKIPPED"
                results.append(result)
                del hidden_states, timestep, encoder_hidden_states
                torch.cuda.empty_cache()
                continue
            else:
                print(
                    f"   Peak: {inference_mem_stats['inference_peak_gb']:.3f} GB | "
                    f"Diff: {inference_mem_stats['inference_mem_diff_gb']:.3f} GB"
                )
                result.update(inference_mem_stats)

            del hidden_states, timestep, encoder_hidden_states
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ Error: {e}")
            result["inference_status"] = "ERROR"
            torch.cuda.empty_cache()

        # 3. 测训练显存 (需要梯度)
        print("🔥 Measuring training memory...")
        try:
            hidden_states, timestep, encoder_hidden_states = create_dummy_inputs(
                transformer, num_frames, requires_grad=True
            )
            training_mem_stats = measure_training_memory(transformer, hidden_states, timestep, encoder_hidden_states)

            if training_mem_stats["status"] == "OOM":
                print("   ❌ OOM")
                result.update(training_mem_stats)
            else:
                print(
                    f"   Peak: {training_mem_stats['training_peak_gb']:.3f} GB | "
                    f"Diff: {training_mem_stats['training_mem_diff_gb']:.3f} GB"
                )
                result.update(training_mem_stats)

            del hidden_states, timestep, encoder_hidden_states
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ Error: {e}")
            result["training_status"] = "ERROR"
            torch.cuda.empty_cache()

        results.append(result)

    return results


# 运行实验
warmup(transformer)

print("\n" + "=" * 80)
print("STANDARD EXPERIMENT")
print("=" * 80)
results_standard = run_experiment(his_tokens, "Standard")

print("\n" + "=" * 80)
print("NAIVE EXPERIMENT")
print("=" * 80)
results_naive = run_experiment(his_tokens_naive, "Naive")

# 保存结果
benchmark_results["experiments"] = [
    {"name": "standard", "results": results_standard},
    {"name": "naive", "results": results_naive},
]

output_file = "benchmark_patchification_results.json"
with open(output_file, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\n" + "=" * 80)
print(f"✅ Results saved to {output_file}")
print("=" * 80)

# 打印汇总表格
print("\n" + "=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)

for exp in benchmark_results["experiments"]:
    print(f"\n=== {exp['name'].upper()} ===")
    print(f"{'Tokens':>6} {'Frames':>6} {'Speed(s)':>10} {'Infer(GB)':>11} {'Train(GB)':>11} {'Status':>10}")
    print("-" * 72)
    for r in exp["results"]:
        speed_str = f"{r.get('avg_time_s', 0):.4f}s" if r.get("status") == "success" else "N/A"
        infer_str = f"{r.get('inference_mem_diff_gb', 0):.3f}" if r.get("inference_peak_gb") else "N/A"
        train_str = f"{r.get('training_mem_diff_gb', 0):.3f}" if r.get("training_peak_gb") else "N/A"

        # 判断整体状态
        if r.get("speed_status") == "OOM":
            status = "OOM"
        elif r.get("training_status") == "OOM":
            status = "OOM(train)"
        elif r.get("status") == "success":
            status = "OK"
        else:
            status = "PARTIAL"

        print(f"{r['his_token']:6d} {r['num_frames']:6d} {speed_str:>10} {infer_str:>11} {train_str:>11} {status:>10}")

print("\n" + "=" * 80)
print("Legend:")
print("  Speed(s)   - Average inference time per step")
print("  Infer(GB)  - Memory usage during inference (forward only)")
print("  Train(GB)  - Memory usage during training (forward + backward)")
print("  Status     - OK/OOM/OOM(train)/PARTIAL")
print("=" * 80)

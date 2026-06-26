import os


os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["DIFFUSERS_ENABLE_HUB_KERNELS"] = "yes"

import json
import time
from datetime import datetime

import torch
from helios.modules.kernels import (
    replace_all_norms_with_flash_norms,
    replace_linear_with_tiled_linear,
    replace_rope_with_flash_rope,
)
from helios.modules.transformer_helios import HeliosTransformer3DModel

from diffusers.training_utils import free_memory


# ============================================================================
# 配置参数
# ============================================================================
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
TEST_NUM_FRAMES = 21
NUM_SPEED_RUNS = 10  # 速度测试的运行次数
HEIGHT = 384
WIDTH = 640

benchmark_results = {
    "timestamp": datetime.now().isoformat(),
    "test_config": {"num_frames": TEST_NUM_FRAMES, "height": HEIGHT, "width": WIDTH, "num_speed_runs": NUM_SPEED_RUNS},
    "experiments": [],
}


# ============================================================================
# 辅助函数
# ============================================================================
def create_dummy_inputs(transformer, num_frames, height=384, width=640, requires_grad=False):
    """创建transformer的dummy输入"""
    batch_size = 1
    device = transformer.device
    dtype = transformer.dtype

    in_channels = transformer.config.in_channels
    latent_h = height // 8
    latent_w = width // 8
    latent_f = num_frames

    hidden_states = torch.randn(
        batch_size, in_channels, latent_f, latent_h, latent_w, device=device, dtype=dtype, requires_grad=requires_grad
    )

    timestep = torch.tensor([999], device=device, dtype=torch.long)
    timestep = timestep.expand(batch_size)

    seq_len = 512
    hidden_dim = 4096
    encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    return hidden_states, timestep, encoder_hidden_states


def measure_inference_speed(transformer, hidden_states, timestep, encoder_hidden_states, num_runs=10):
    """测量推理速度"""
    try:
        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
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
                    return_dict=False,
                )[0]

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
            free_memory()
            return {"status": "OOM", "error": str(e)}
        else:
            raise


def measure_inference_memory(transformer, hidden_states, timestep, encoder_hidden_states):
    """测量推理显存"""
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        free_memory()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
                attention_kwargs=None,
            )[0]
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
            free_memory()
            return {"status": "OOM", "error": str(e)}
        else:
            raise


def measure_training_speed(transformer, hidden_states, timestep, encoder_hidden_states, num_runs=10):
    """测量训练速度(forward + backward)"""
    try:
        # 预热
        for _ in range(3):
            output = transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            loss = output.sum()
            loss.backward()
            transformer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # 正式测速
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()

            output = transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            loss = output.sum()
            loss.backward()
            transformer.zero_grad(set_to_none=True)

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
            free_memory()
            transformer.zero_grad(set_to_none=True)
            return {"status": "OOM", "error": str(e)}
        else:
            raise


def measure_training_memory(transformer, hidden_states, timestep, encoder_hidden_states):
    """测量训练显存(forward + backward)"""
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        free_memory()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3

        torch.cuda.reset_peak_memory_stats()

        output = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
            attention_kwargs=None,
        )[0]

        loss = output.sum()
        loss.backward()

        torch.cuda.synchronize()

        training_peak = torch.cuda.max_memory_allocated() / 1024**3
        training_mem_diff = training_peak - mem_before

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
            free_memory()
            transformer.zero_grad(set_to_none=True)
            return {"status": "OOM", "error": str(e)}
        else:
            raise


def run_single_config(transformer, config_name, num_frames):
    """运行单个配置的完整测试"""
    print(f"\n{'=' * 70}")
    print(f"Testing: {config_name}")
    print(f"{'=' * 70}")

    result = {"config": config_name, "num_frames": num_frames}

    # 1. 测推理速度
    print("📊 Measuring inference speed...")
    try:
        hidden_states, timestep, encoder_hidden_states = create_dummy_inputs(
            transformer, num_frames, HEIGHT, WIDTH, requires_grad=False
        )
        speed_stats = measure_inference_speed(
            transformer, hidden_states, timestep, encoder_hidden_states, NUM_SPEED_RUNS
        )

        if speed_stats["status"] == "OOM":
            print("   ❌ OOM - Skipping remaining tests")
            result.update(
                {
                    "inference_speed_status": "OOM",
                    "inference_memory_status": "SKIPPED",
                    "training_speed_status": "SKIPPED",
                    "training_memory_status": "SKIPPED",
                }
            )
            del hidden_states, timestep, encoder_hidden_states
            torch.cuda.empty_cache()
            free_memory()
            return result
        else:
            print(
                f"   ✓ Avg: {speed_stats['avg_time_s']:.4f}s | "
                f"Min: {speed_stats['min_time_s']:.4f}s | "
                f"Max: {speed_stats['max_time_s']:.4f}s"
            )
            result.update(
                {
                    "inference_speed_avg_s": speed_stats["avg_time_s"],
                    "inference_speed_min_s": speed_stats["min_time_s"],
                    "inference_speed_max_s": speed_stats["max_time_s"],
                    "inference_speed_std_s": speed_stats["std_time_s"],
                    "inference_speed_status": "success",
                }
            )

        del hidden_states, timestep, encoder_hidden_states
        torch.cuda.empty_cache()
        free_memory()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        result["inference_speed_status"] = "ERROR"
        torch.cuda.empty_cache()
        free_memory()

    # 2. 测推理显存
    print("💾 Measuring inference memory...")
    try:
        hidden_states, timestep, encoder_hidden_states = create_dummy_inputs(
            transformer, num_frames, HEIGHT, WIDTH, requires_grad=False
        )
        mem_stats = measure_inference_memory(transformer, hidden_states, timestep, encoder_hidden_states)

        if mem_stats["status"] == "OOM":
            print("   ❌ OOM - Skipping training tests")
            result.update(
                {
                    "inference_memory_status": "OOM",
                    "training_speed_status": "SKIPPED",
                    "training_memory_status": "SKIPPED",
                }
            )
            del hidden_states, timestep, encoder_hidden_states
            torch.cuda.empty_cache()
            free_memory()
            return result
        else:
            print(
                f"   ✓ Peak: {mem_stats['inference_peak_gb']:.3f} GB | "
                f"Diff: {mem_stats['inference_mem_diff_gb']:.3f} GB"
            )
            result.update(
                {
                    "inference_memory_peak_gb": mem_stats["inference_peak_gb"],
                    "inference_memory_diff_gb": mem_stats["inference_mem_diff_gb"],
                    "inference_memory_status": "success",
                }
            )

        del hidden_states, timestep, encoder_hidden_states
        torch.cuda.empty_cache()
        free_memory()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        result["inference_memory_status"] = "ERROR"
        torch.cuda.empty_cache()
        free_memory()

    # 3. 测训练速度
    print("⚡ Measuring training speed...")
    try:
        hidden_states, timestep, encoder_hidden_states = create_dummy_inputs(
            transformer, num_frames, HEIGHT, WIDTH, requires_grad=True
        )
        train_speed_stats = measure_training_speed(
            transformer, hidden_states, timestep, encoder_hidden_states, NUM_SPEED_RUNS
        )

        if train_speed_stats["status"] == "OOM":
            print("   ❌ OOM")
            result.update({"training_speed_status": "OOM", "training_memory_status": "SKIPPED"})
            del hidden_states, timestep, encoder_hidden_states
            torch.cuda.empty_cache()
            free_memory()
            return result
        else:
            print(
                f"   ✓ Avg: {train_speed_stats['avg_time_s']:.4f}s | "
                f"Min: {train_speed_stats['min_time_s']:.4f}s | "
                f"Max: {train_speed_stats['max_time_s']:.4f}s"
            )
            result.update(
                {
                    "training_speed_avg_s": train_speed_stats["avg_time_s"],
                    "training_speed_min_s": train_speed_stats["min_time_s"],
                    "training_speed_max_s": train_speed_stats["max_time_s"],
                    "training_speed_std_s": train_speed_stats["std_time_s"],
                    "training_speed_status": "success",
                }
            )

        del hidden_states, timestep, encoder_hidden_states
        torch.cuda.empty_cache()
        free_memory()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        result["training_speed_status"] = "ERROR"
        torch.cuda.empty_cache()
        free_memory()

    # 4. 测训练显存
    print("🔥 Measuring training memory...")
    try:
        hidden_states, timestep, encoder_hidden_states = create_dummy_inputs(
            transformer, num_frames, HEIGHT, WIDTH, requires_grad=True
        )
        train_mem_stats = measure_training_memory(transformer, hidden_states, timestep, encoder_hidden_states)

        if train_mem_stats["status"] == "OOM":
            print("   ❌ OOM")
            result["training_memory_status"] = "OOM"
        else:
            print(
                f"   ✓ Peak: {train_mem_stats['training_peak_gb']:.3f} GB | "
                f"Diff: {train_mem_stats['training_mem_diff_gb']:.3f} GB"
            )
            result.update(
                {
                    "training_memory_peak_gb": train_mem_stats["training_peak_gb"],
                    "training_memory_diff_gb": train_mem_stats["training_mem_diff_gb"],
                    "training_memory_status": "success",
                }
            )

        del hidden_states, timestep, encoder_hidden_states
        torch.cuda.empty_cache()
        free_memory()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        result["training_memory_status"] = "ERROR"
        torch.cuda.empty_cache()
        free_memory()

    return result


# ============================================================================
# 主测试流程
# ============================================================================
print("=" * 70)
print("OPTIMIZATION BENCHMARK - SAME LENGTH COMPARISON")
print("=" * 70)
print(f"Model: {model_id}")
print(f"Test frames: {TEST_NUM_FRAMES}")
print(f"Resolution: {HEIGHT}x{WIDTH}")
print(f"Speed test runs: {NUM_SPEED_RUNS}")
print("=" * 70)

# ============================================================================
# 配置1: 原始模型
# ============================================================================
print("\n" + "=" * 70)
print("CONFIG 1/5: BASELINE (No optimizations)")
print("=" * 70)

transformer_baseline = HeliosTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_default_loader=True,
)
transformer_baseline.enable_gradient_checkpointing()
transformer_baseline.set_attention_backend("_flash_3_hub")
transformer_baseline.to("cuda")

result_baseline = run_single_config(transformer_baseline, "Baseline", TEST_NUM_FRAMES)
benchmark_results["experiments"].append(result_baseline)

del transformer_baseline
torch.cuda.empty_cache()
free_memory()

# ============================================================================
# 配置2: 只替换 TiledLinear
# ============================================================================
print("\n" + "=" * 70)
print("CONFIG 2/5: TiledLinear only")
print("=" * 70)

transformer_tiled = HeliosTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_default_loader=True,
)
transformer_tiled.enable_gradient_checkpointing()
transformer_tiled.set_attention_backend("_flash_3_hub")
transformer_tiled = replace_linear_with_tiled_linear(transformer_tiled)
transformer_tiled.to("cuda")

result_tiled = run_single_config(transformer_tiled, "TiledLinear", TEST_NUM_FRAMES)
benchmark_results["experiments"].append(result_tiled)

transformer_tiled = None
del transformer_tiled
torch.cuda.empty_cache()
free_memory()

# ============================================================================
# 配置3: 只替换 FlashNorm
# ============================================================================
print("\n" + "=" * 70)
print("CONFIG 3/5: FlashNorm only")
print("=" * 70)

transformer_flashnorm = HeliosTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_default_loader=True,
)
transformer_flashnorm.enable_gradient_checkpointing()
transformer_flashnorm.set_attention_backend("_flash_3_hub")
transformer_flashnorm = replace_all_norms_with_flash_norms(transformer_flashnorm)
transformer_flashnorm.to("cuda")

result_flashnorm = run_single_config(transformer_flashnorm, "FlashNorm", TEST_NUM_FRAMES)
benchmark_results["experiments"].append(result_flashnorm)

transformer_flashnorm = None
del transformer_flashnorm
torch.cuda.empty_cache()
free_memory()

# ============================================================================
# 配置4: 只替换 FlashRoPE
# ============================================================================
print("\n" + "=" * 70)
print("CONFIG 4/5: FlashRoPE only")
print("=" * 70)

transformer_flashrope = HeliosTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_default_loader=True,
)
transformer_flashrope.enable_gradient_checkpointing()
transformer_flashrope.set_attention_backend("_flash_3_hub")
transformer_flashrope.to("cuda")

# FlashRoPE 是全局替换,不可逆
replace_rope_with_flash_rope()

result_flashrope = run_single_config(transformer_flashrope, "FlashRoPE", TEST_NUM_FRAMES)
benchmark_results["experiments"].append(result_flashrope)

transformer_flashrope = None
del transformer_flashrope
torch.cuda.empty_cache()
free_memory()

# ============================================================================
# 配置5: FlashNorm + FlashRoPE
# ============================================================================
print("\n" + "=" * 70)
print("CONFIG 5/5: FlashNorm + FlashRoPE")
print("=" * 70)

transformer_combined = HeliosTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_default_loader=True,
)
transformer_combined.enable_gradient_checkpointing()
transformer_combined.set_attention_backend("_flash_3_hub")
transformer_combined = replace_all_norms_with_flash_norms(transformer_combined)
transformer_combined.to("cuda")

# FlashRoPE 已经在配置4中全局替换
replace_rope_with_flash_rope()

result_combined = run_single_config(transformer_combined, "FlashNorm+FlashRoPE", TEST_NUM_FRAMES)
benchmark_results["experiments"].append(result_combined)

transformer_combined = None
del transformer_combined
torch.cuda.empty_cache()
free_memory()

# ============================================================================
# 保存结果
# ============================================================================
output_file = "benchmark_triton_results.json"
with open(output_file, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\n" + "=" * 70)
print(f"✅ Results saved to {output_file}")
print("=" * 70)

# ============================================================================
# 打印汇总表格
# ============================================================================
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

# 表头
print(f"\n{'Config':<20} {'InfSpeed(s)':>12} {'InfMem(GB)':>12} {'TrainSpeed(s)':>14} {'TrainMem(GB)':>13}")
print("-" * 75)

# 打印每个配置的结果
for exp in benchmark_results["experiments"]:
    config = exp["config"]

    # 推理速度
    inf_speed = (
        f"{exp.get('inference_speed_avg_s', 0):.4f}" if exp.get("inference_speed_status") == "success" else "N/A"
    )

    # 推理显存
    inf_mem = (
        f"{exp.get('inference_memory_diff_gb', 0):.3f}" if exp.get("inference_memory_status") == "success" else "N/A"
    )

    # 训练速度
    train_speed = (
        f"{exp.get('training_speed_avg_s', 0):.4f}" if exp.get("training_speed_status") == "success" else "N/A"
    )

    # 训练显存
    train_mem = (
        f"{exp.get('training_memory_diff_gb', 0):.3f}" if exp.get("training_memory_status") == "success" else "N/A"
    )

    print(f"{config:<20} {inf_speed:>12} {inf_mem:>12} {train_speed:>14} {train_mem:>13}")

# 计算加速比(如果baseline成功)
baseline_result = benchmark_results["experiments"][0]
if baseline_result.get("inference_speed_status") == "success":
    baseline_inf_speed = baseline_result["inference_speed_avg_s"]
    baseline_train_speed = baseline_result.get("training_speed_avg_s", None)

    print("\n" + "=" * 70)
    print("SPEEDUP vs BASELINE")
    print("=" * 70)
    print(f"{'Config':<20} {'InfSpeedup':>12} {'TrainSpeedup':>14}")
    print("-" * 50)

    for exp in benchmark_results["experiments"]:
        config = exp["config"]

        # 推理加速比
        if exp.get("inference_speed_status") == "success":
            speedup_inf = baseline_inf_speed / exp["inference_speed_avg_s"]
            speedup_inf_str = f"{speedup_inf:.2f}x"
        else:
            speedup_inf_str = "N/A"

        # 训练加速比
        if exp.get("training_speed_status") == "success" and baseline_train_speed:
            speedup_train = baseline_train_speed / exp["training_speed_avg_s"]
            speedup_train_str = f"{speedup_train:.2f}x"
        else:
            speedup_train_str = "N/A"

        print(f"{config:<20} {speedup_inf_str:>12} {speedup_train_str:>14}")

print("\n" + "=" * 70)
print("Legend:")
print("  InfSpeed   - Inference time (forward only)")
print("  InfMem     - Inference memory usage")
print("  TrainSpeed - Training time (forward + backward)")
print("  TrainMem   - Training memory usage")
print("  Speedup    - Relative to baseline (higher is better)")
print("=" * 70)

import os
import sys
import time
from datetime import datetime

import torch


os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["DIFFUSERS_ENABLE_HUB_KERNELS"] = "yes"

from helios.modules.kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_wan import WanPipeline

from diffusers import AutoencoderKLWan


class DualLogger:
    """同时输出到控制台和文件的日志器"""

    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)  # 输出到控制台
        self.file.write(message)  # 写入文件
        self.file.flush()  # 实时刷新

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def setup_pipeline(model_id, compile_config=None):
    """设置pipeline"""
    print(f"\n{'=' * 60}")
    print(f"设置 Pipeline: {compile_config['name'] if compile_config else 'No Compile'}")
    print(f"{'=' * 60}")

    # 加载模型
    transformer = HeliosTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16, use_default_loader=True
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16)

    pipe.transformer.set_attention_backend("_flash_3_hub")
    pipe.to("cuda")

    # 应用compile配置
    if compile_config:
        print(f"应用编译配置: {compile_config['kwargs']}")
        pipe.transformer.compile(**compile_config["kwargs"])

    return pipe


def run_benchmark(pipe, prompt, negative_prompt, num_runs=3, warmup=1):
    """运行基准测试"""
    times = []

    # Warmup
    print(f"\n预热运行 {warmup} 次...")
    for i in range(warmup):
        print(f"  预热 {i + 1}/{warmup}")
        _ = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=384,
            width=640,
            num_frames=45,
            guidance_scale=5.0,
            num_inference_steps=50,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        torch.cuda.empty_cache()

    # 实际测试
    print(f"\n开始基准测试 {num_runs} 次...")
    for i in range(num_runs):
        print(f"  运行 {i + 1}/{num_runs}")
        start = time.time()
        torch.cuda.synchronize()

        _ = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=384,
            width=640,
            num_frames=45,
            guidance_scale=5.0,
            num_inference_steps=50,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    耗时: {elapsed:.2f}秒")
        torch.cuda.empty_cache()

    return times


def main():
    # 创建日志文件
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = "benchmark_compile_results.txt"

    # 创建双输出日志器
    logger = DualLogger(log_filename)
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        # 打印测试信息
        print("=" * 80)
        print("PyTorch Compile 模式基准测试")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 80)

        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

        prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        # 定义不同的compile配置
        compile_configs = [
            {"name": "No Compile (Baseline)", "kwargs": None},
            {"name": "Default Compile", "kwargs": {}},
            {"name": "Fullgraph Only", "kwargs": {"fullgraph": True}},
            {
                "name": "Max-Autotune-No-Cudagraphs + Dynamic",
                "kwargs": {"mode": "max-autotune-no-cudagraphs", "dynamic": True},
            },
            {"name": "Max-Autotune + Fullgraph", "kwargs": {"mode": "max-autotune", "fullgraph": True}},
            {"name": "Max-Autotune", "kwargs": {"mode": "max-autotune"}},
            {"name": "Reduce-Overhead", "kwargs": {"mode": "reduce-overhead"}},
            {"name": "Default Mode", "kwargs": {"mode": "default"}},
        ]

        results = {}

        # 测试每个配置
        for config in compile_configs:
            try:
                # 清理GPU内存
                torch.cuda.empty_cache()

                # 设置pipeline
                if config["kwargs"] is None:
                    pipe = setup_pipeline(model_id, None)
                else:
                    pipe = setup_pipeline(model_id, config)

                # 运行基准测试
                times = run_benchmark(pipe, prompt, negative_prompt, num_runs=3, warmup=1)
                results[config["name"]] = times

                # 删除pipeline释放内存
                del pipe
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n❌ 配置 '{config['name']}' 失败: {str(e)}")
                results[config["name"]] = None

        # 打印结果摘要
        print("\n" + "=" * 80)
        print("基准测试结果摘要")
        print("=" * 80)
        print(f"{'配置':<45} {'平均时间(秒)':<15} {'最小时间(秒)':<15} {'最大时间(秒)':<15}")
        print("-" * 80)

        sorted_results = []
        for name, times in results.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                sorted_results.append((name, avg_time, min_time, max_time, times))
                print(f"{name:<45} {avg_time:<15.2f} {min_time:<15.2f} {max_time:<15.2f}")
            else:
                print(f"{name:<45} {'FAILED':<15} {'FAILED':<15} {'FAILED':<15}")

        # 按平均时间排序
        if sorted_results:
            sorted_results.sort(key=lambda x: x[1])
            print("\n" + "=" * 80)
            print("速度排名 (从快到慢)")
            print("=" * 80)
            baseline_time = sorted_results[-1][1]
            for rank, (name, avg_time, min_time, max_time, times) in enumerate(sorted_results, 1):
                speedup = baseline_time / avg_time if avg_time > 0 else 0
                print(f"\n{rank}. {name}")
                print(f"   平均时间: {avg_time:.2f}秒")
                print(f"   相对最慢提速: {speedup:.2f}x")
                print(f"   详细时间: {[f'{t:.2f}s' for t in times]}")

        print("\n" + "=" * 80)
        print(f"测试完成! 结果已保存到: {log_filename}")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试过程出错: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # 恢复标准输出并关闭文件
        sys.stdout = original_stdout
        logger.close()
        print(f"\n✅ 测试完成! 结果已保存到: {log_filename}")


if __name__ == "__main__":
    main()

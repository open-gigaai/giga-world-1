# Radial Attention

### [Paper](https://arxiv.org/abs/2506.19852) | [Website](https://hanlab.mit.edu/projects/radial-attention)

**[2025-10-29]** Radial+SageAttention2++ is now supported on RTX-5090 GPU!

**[2025-09-18]** We are thrilled to announce that Radial Attention is **accepted by NeurIPS 2025!** 🎉🎉🎉 Please feel free to reach out at San Diego!

**[2025-09-04]** Radial Attention now supports multi-gpu inference using ulysses sequence from [xDiT project](https://github.com/xdit-project/xDiT/tree/main), great thanks to [Zheming](https://github.com/1145284121) for making this important PR.

**[2025-08-04]** Radial Attention now supports [Lightx2v](https://github.com/ModelTC/LightX2V), a 4-step LoRA. Radial Attention also supports [SageAttention2++](https://arxiv.org/abs/2505.21136) for FP8 Matmul accumulation on 4090. With the joint effort of **Radial Attention, SageAttention and Lightx2v LoRA**, now it only takes **33/90 seconds to generate a high-fidelity video for Wan2.1 on a single H100/4090 GPU respectively!**.

**[2025-07-22]** Radial Attention is now compatible with [SageAttention](https://github.com/thu-ml/SageAttention) version 2!

**[2025-07-14]** Radial Attention is now compatible with [SageAttention](https://github.com/thu-ml/SageAttention) version 1!

**[2025-07-03]** Radial Attention now supports [Wan2.1_14B_FusionX LoRA](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX)! You can get high-quality videos within just 8 steps (90 seconds on a single H100 GPU)!

**[2025-06-24]** Radial Attention is open-sourced! Wan2.1-14B, HunyuanVideo, and Mochi-1 are supported for fast video generation with high quality under 1-4⨉ video length.

https://github.com/user-attachments/assets/af1aaf29-4123-4e4c-9c1a-4360f63d7ce0

We present *Radial Attention*, a sparse attention mechanism with $\mathcal{O}(n\log n)$ computational complexity. Radial Attention accelerates pre-trained HunyuanVideo by 1.9× at its default video length while maintaining comparable video quality. When generating 4× longer videos, it reduces tuning costs by up to 4.4× and speeds up inference by up to 3.7× versus dense attention.

**Radial Attention: $\mathcal{O}(n\log n)$ Sparse Attention with Energy Decay for Long Video Generation**

[Xingyang Li](https://acm.sjtu.edu.cn/~xyli)\*, [Muyang Li](https://lmxyy.me/)\*, [Tianle Cai](https://www.tianle.website/#/), [Haocheng Xi](https://haochengxi.github.io/), [Shuo Yang](https://andy-yang-1.github.io/), [Yujun Lin](https://yujunlin.com/), [Lvmin Zhang](https://scholar.google.com/citations?user=ANMsdHYAAAAJ&hl=en), [Songlin Yang](https://sustcsonglin.github.io/), Jinbo Hu, Kelly Peng, [Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/), [Ion Stoica](https://people.eecs.berkeley.edu/~istoica/), [Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/), and [Song Han](https://hanlab.mit.edu/songhan)

MIT, NVIDIA, Princeton, UC Berkeley, Stanford, and First Intelligence

## 📖Overview

![teaser](https://github.com/user-attachments/assets/aa69414b-8d7e-4ba5-9b9f-9dcb4bb3cf90)

**Radial Attention** is a **scalable sparse attention mechanism** for video diffusion models that translates **Spatiotemporal Energy Decay**—observed in attention score distributions—into exponentially decaying compute density. Unlike $\mathcal{O}(n^2)$ dense attention  or linear approximations, Radial Attention achieves **$\mathcal{O}(n \log n)$ complexity** while preserving expressive power for long videos. Here are our core contributions.

- **Physics-Inspired Sparsity**: Static masks enforce *spatially local* and *temporally decaying* attention, mirroring energy dissipation in physical systems.
- **Efficient Length Extension**: Pre-trained models (e.g., Wan2.1-14B, HunyuanVideo) scale to **4× longer videos** via lightweight LoRA tuning, avoiding full-model retraining.

## 🔍Sparsity Pattern Design

![patterns](https://github.com/user-attachments/assets/8e572cc5-27f3-4b24-bc0e-7d0a9d0b3cde)

**(a)** The compute density pattern. The attention map is divided into $2\lceil\log_2(\max(f, 2))\rceil - 1$ bands (here, the number of frames $f = 12$) based on the temporal distance between tokens. The central band has full compute density, while each successive outer band has half the density of the previous one. Except for band $\pm1$, each band also doubles the diagonal width of its predecessor.  
**(b)** The corresponding attention mask for (a). The compute density is reflected in the compute diagonal width of each frame-to-frame block. When the diagonal width drops below 1, we reduce the frequency of diagonals. We additionally add an attention sink.  
**(c)** An example mask used in HunyuanVideo, illustrating the final sparsity pattern in practice.

## 📊Performance

![results](https://github.com/user-attachments/assets/861ffe21-3365-4bf3-abb1-852d4f20bc8d)

**Radial Attention** reduces the computational complexity of attention from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$. When generating a 500-frame 720p video with HunyuanVideo, it reduces the attention computation by 9×, achieves 3.7× speedup, and saves 4.6× tuning costs.

## 🎥Visual Results
### 🔹Accelerating Pre-trained Models
![image](https://github.com/user-attachments/assets/ad488f95-a02e-4b62-a107-1bed40623a24)
Radial Attention delivers nearly identical quality to Wan2.1-14B at default video length, while offering **1.8× speedup**.

### 🔹Long Video Generation
![image](https://github.com/user-attachments/assets/0d3cecb3-2f45-4a12-b1ba-e4a398628e22)
Radial Attention enables **4× longer video generation** with LoRA tuning, outperforming dense attention in vision rewards, while achieving **3.7× speedup** and **4.4× lower tuning costs**.

### 🔹LoRA Compatibility
![image](https://github.com/user-attachments/assets/9aaab627-a8cc-4132-a801-0432e3d8d764)
Fully compatible with existing style LoRAs. On HunyuanVideo, Radial Attention LoRA enables 4× video length extension while preserving vision quality.

### 🔹LoRA

## 🔧Installation

We start with cloning the repository:

```bash
git clone git@github.com:mit-han-lab/radial-attention
cd radial-attention

# Initialize submodules with custom patches
bash scripts/init_submodules.sh
```

We recommend using CUDA versions 12.4 + Pytorch versions 2.5.1

```bash
# 1. Create and activate conda environment
conda create -n radial python==3.12 -y
conda activate radial

# 2. Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Install pip dependencies from CogVideoX and HunyuanVideo
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# 4. Install FlashInfer for fast and hardware-friendly inference
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/

# 5. Install Latest Diffusers to try lightx2v features and Wan2.2
pip install git+https://github.com/huggingface/diffusers

# 6. (Optional) Install Sparse_SageAttention for further acceleration
cd third_party/SageAttention/ # install SageAttention
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # parallel compiling (Optional)
python setup.py install  # or pip install -e .
cd ../..
cd third_party/sparse_sageattn # if you want to use Radial Attention with SageAttention v1 backend
python setup.py install
cd ../..
cd third_party/sparse_sageattn_2 # if you want to use Radial Attention with SageAttention v2 backend
pip install ninja   # for parallel compilation
python setup.py install   # or pip install -e .
cd ../..
```

## 🚀Inference Examples

### Wan2.1-14B

We support Text-to-Video inference of Wan2.1-14B. The running script is:

```bash
bash scripts/wan_t2v_inference.sh
```

### HunyuanVideo

We support Text-to-Video inference of HunyuanVideo. The running script is:

```bash
bash scripts/hunyuan_t2v_inference.sh
```

## 📕Open-source Plan

- [x] Integrate [Wan2.1_14B_FusionX LoRA](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX) for high-quality few-step generation
- [ ] Adopt [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen/tree/main)'s fused kernels for further speedup
- [x] ComfyUI integration (in [ComfyUI-nunchaku](https://github.com/mit-han-lab/ComfyUI-nunchaku))
- [ ] Support Mochi-1
- [x] Support Multi-GPU inference
- [ ] Release LoRA checkpoints for longer-video generation

## 📚Citation

If you find Radial Attention useful or relevant to your research, please cite our paper:

```bibtex
@article{li2025radial,
  title={Radial Attention: $\mathcal{O}(n\log n)$ Sparse Attention with Energy Decay for Long Video Generation},
  author={Li*, Xingyang and Li*, Muyang and Cai, Tianle and Xi, Haocheng and Yang, Shuo and Lin, Yujun and Zhang, Lvmin and Yang, Songlin and Hu, Jinbo and Peng, Kelly and Agrawala, Maneesh and Stoica, Ion and Keutzer, Kurt and Han, Song},
  journal={arXiv preprint arXiv:2506.19852},
  year={2025}
}
```

## Acknowledgements

We thank [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen/tree/main) for insights on code design.

We thank MIT-IBM Watson AI Lab, National Science Foundation, Hyundai, and Amazon for supporting this research.

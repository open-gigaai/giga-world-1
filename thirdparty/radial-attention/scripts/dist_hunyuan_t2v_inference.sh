dense_layers=0
dense_timesteps=10

prompt=$(cat examples/prompt.txt)

CUDA_VISIBLE_DEVICES=5,3,6,7 torchrun --nproc_per_node=4 \
  hunyuan_t2v_inference.py \
    --prompt "$prompt" \
    --height 768 \
    --width 1280 \
    --num_frames 117 \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --decay_factor 0.95 \
    --use_sequence_parallel \
    --ulysses_degree 4 \
    --num_inference_steps 50 \
    --pattern "radial" \
    --output_file "hunyuan_radial_sp4.mp4"
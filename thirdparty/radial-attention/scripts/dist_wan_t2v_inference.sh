# this is the setting for 1x length T2V inference
dense_layers=1
dense_timesteps=10

prompt=$(cat examples/prompt.txt)

CUDA_VISIBLE_DEVICES=5,3,6,7 torchrun --nproc_per_node=4 \
    wan_t2v_inference.py \
    --prompt "$prompt" \
    --height 768 \
    --width 1280 \
    --num_frames 69 \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --decay_factor 0.2 \
    --use_sequence_parallel \
    --ulysses_degree 4 \
    --num_inference_steps 50 \
    --pattern "radial" \
    --output_file "wan_radial_sp4.mp4"
# this is the setting for 1x length T2V inference
dense_layers=1
dense_timesteps=12

prompt=$(cat examples/prompt.txt)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python wan_t2v_inference.py \
    --prompt "$prompt" \
    --height 768 \
    --width 1280 \
    --num_frames 69 \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --decay_factor 0.2 \
    --pattern "radial" \
    --use_sage_attention \
    --num_inference_steps 50 \
    --use_model_offload \
    --output_file "output_768_radial_sage.mp4"
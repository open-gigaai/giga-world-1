dense_layers=0
dense_timesteps=12

prompt=$(cat examples/prompt.txt)

python hunyuan_t2v_inference.py \
    --prompt "$prompt" \
    --height 720 \
    --width 1280 \
    --num_frames 125 \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --decay_factor 0.95 \
    --pattern "radial" \
    --output_file "output_320_radial_sage.mp4" \
    --use_model_offload \
    --num_inference_steps 50 \
    --use_sage_attention \
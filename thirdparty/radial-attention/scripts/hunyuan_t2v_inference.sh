dense_layers=0
dense_timesteps=12

prompt=$(cat examples/prompt.txt)

python hunyuan_t2v_inference.py \
    --prompt "$prompt" \
    --height 768 \
    --width 1280 \
    --num_frames 117 \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --decay_factor 0.95 \
    --pattern "radial" \
    --output_file "output_radial.mp4"

# this is the setting for 4x length T2V inference
# dense_layers=2
# dense_timesteps=2
# prompt=$(cat examples/prompt.txt)
# python hunyuan_t2v_inference.py \
#     --prompt "$prompt" \
#     --height 720 \
#     --width 1280 \
#     --num_frames 509 \
#     --pattern "radial" \
#     --dense_layers $dense_layers \
#     --dense_timesteps $dense_timesteps \
#     --lora_checkpoint_dir data/hunyuan_4x_lora \
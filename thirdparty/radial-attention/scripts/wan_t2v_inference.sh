# this is the setting for 1x length T2V inference
dense_layers=1
dense_timesteps=12

prompt=$(cat examples/prompt.txt)

python wan_t2v_inference.py \
    --prompt "$prompt" \
    --height 768 \
    --width 1280 \
    --num_frames 69 \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --decay_factor 0.2 \
    --pattern "radial"

# this is the setting for 2x length T2V inference
# dense_layers=2
# dense_timesteps=2

# python wan_t2v_inference.py \
#     --prompt "$prompt" \
#     --height 720 \
#     --width 1280 \
#     --num_frames 161 \
#     --pattern "radial" \
#     --dense_layers $dense_layers \
#     --dense_timesteps $dense_timesteps \
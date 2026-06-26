#!/bin/bash

# Wan2.2 T2V inference script with radial attention
# Default configuration for high-quality video generation

dense_layers=1
dense_timesteps=11
decay_factor=0.8

prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

python wan_22_t2v_inference.py \
    --prompt "$prompt" \
    --height 768 \
    --width 1280 \
    --num_frames 77 \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --decay_factor $decay_factor \
    --pattern "radial" \
    --guidance_scale 4.0 \
    --guidance_scale_2 3.0 \
    --num_inference_steps 40 \
    --output_file "wan22_radial_output.mp4" \
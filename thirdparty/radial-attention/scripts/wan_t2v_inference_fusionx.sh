# this is the setting for 1x length T2V inference
dense_layers=1
dense_timesteps=2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

prompt=$(cat examples/prompt.txt)

python wan_t2v_inference.py \
    --prompt "$prompt" \
    --height 768 \
    --width 1280 \
    --num_frames 69 \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --decay_factor 0.2 \
    --pattern "radial" \
    --num_inference_steps 8 \
    --output_file "gt_fusionx.mp4" \
    --guidance_scale 1.0 \
    --flow_shift 2.0 \
    --lora_checkpoint_dir "vrgamedevgirl84/Wan14BT2VFusioniX" \
    --lora_checkpoint_name "FusionX_LoRa/Wan2.1_T2V_14B_FusionX_LoRA.safetensors" \
    --prompt "A dense fog rolls over an ancient forest at dusk. The camera slowly tracks forward through the mist, revealing tall, gnarled trees draped in moss. Soft, diffused light filters through the fog, casting long shadows on the ground. The atmosphere is eerie and haunting, with a faint glow from the horizon. A low-angle shot captures the towering trees against a darkening sky, evoking a sense of mystery and foreboding. The scene feels still and ancient, yet alive with secrets." \
    --use_sage_attention \
    --use_model_offload \

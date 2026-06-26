# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_gigaworld.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

CUDA_VISIBLE_DEVICES=0 python infer_gigaworld.py \
    --base_model_path "BestWishYsh/Gigaworld-Distilled" \
    --transformer_path "BestWishYsh/Gigaworld-Distilled" \
    --sample_type "i2v" \
    --num_frames 240 \
    --fps 24 \
    --image_path "example/wave.jpg" \
    --image_noise_sigma_min 0.111 \
    --image_noise_sigma_max 0.135 \
    --prompt "A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, illuminating the intricate textures and deep green hues within the wave’s body. A thick spray erupts from the breaking crest, casting a misty veil that dances above the churning surface. As the perspective widens, the immense scale of the wave becomes apparent, revealing the restless expanse of the ocean stretching beyond. The scene captures the ocean’s untamed beauty and relentless force, with every droplet and ripple shimmering in the light. The dynamic motion and vivid colors evoke both awe and respect for nature’s might." \
    --guidance_scale 1.0 \
    --is_amplify_first_chunk \
    --enable_compile \
    --output_folder "./output_gigaworld/gigaworld-distilled"


    # --enable_low_vram_mode \
    # --group_offloading_type "leaf_level" \  # ["leaf_level", "block_level"]
    # --num_blocks_per_group 4 \
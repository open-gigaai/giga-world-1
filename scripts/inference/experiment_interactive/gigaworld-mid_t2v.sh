# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_gigaworld.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

CUDA_VISIBLE_DEVICES=0 python infer_gigaworld.py \
    --base_model_path "BestWishYsh/Gigaworld-Mid" \
    --transformer_path "BestWishYsh/Gigaworld-Mid" \
    --sample_type "t2v" \
    --num_frames 1452 \
    --fps 24 \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement." \
    --guidance_scale 5.0 \
    --use_zero_init \
    --zero_steps 1 \
    --enable_compile \
    --interpolation_steps 3 \
    --interactive_prompt_csv_path "example/prompt_interactive_gigaworld.csv" \
    --interpolate_time 7 \
    --output_folder "./output_gigaworld/gigaworld-mid"


    # --enable_low_vram_mode \
    # --group_offloading_type "leaf_level" \  # ["leaf_level", "block_level"]
    # --num_blocks_per_group
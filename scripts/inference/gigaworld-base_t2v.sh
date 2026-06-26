# # Example: Running inference with 2-GPU parallelism
# # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_gigaworld.py \
# #     --enable_parallelism \
# #     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

# # Ensure the Gigaworld conda env is active so that the local editable
# # `diffusers` (thirdparty/diffusers) under /mnt/pfs/users/zhanqian.wu/env/Gigaworld
# # is used instead of the system diffusers in /opt/conda.
# if [ -z "${CONDA_PREFIX}" ] || [ "${CONDA_PREFIX}" != "/mnt/pfs/users/zhanqian.wu/env/Gigaworld" ]; then
#     source /mnt/pfs/users/zhanqian.wu/miniconda3/etc/profile.d/conda.sh
#     conda activate /mnt/pfs/users/zhanqian.wu/env/Gigaworld
# fi

# CUDA_VISIBLE_DEVICES=0 python infer_gigaworld.py \
#     --base_model_path "/mnt/pfs/users/zhanqian.wu/ckpt/gigaworld_14b" \
#     --transformer_path "/mnt/pfs/users/zhanqian.wu/ckpt/gigaworld_14b" \
#     --sample_type "t2v" \
#     --num_frames 99 \
#     --fps 24 \
#     --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement." \
#     --guidance_scale 5.0 \
#     --enable_compile \
#     --output_folder "./output_gigaworld/gigaworld-base"

#     # --enable_low_vram_mode \
#     # --group_offloading_type "leaf_level" \  # ["leaf_level", "block_level"]
#     # --num_blocks_per_group 4 \
#     # --use_cfg_zero_star \
#     # --use_zero_init \
#     # --zero_steps 1 \

# 🔥 4卡并行 + 极速加速（你真正该用的）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 infer_gigaworld.py \
    --enable_parallelism \
    --cp_backend "ulysses" \
    --base_model_path "/mnt/pfs/users/zhanqian.wu/ckpt/gigaworld_14b" \
    --transformer_path "/mnt/pfs/users/zhanqian.wu/ckpt/gigaworld_14b" \
    --sample_type "t2v" \
    --num_frames 99 \
    --fps 24 \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean." \
    --guidance_scale 5.0 \
    --output_folder "./output_gigaworld/gigaworld-8gpu"\
    --enable_compile \
    --use_zero_init \
    --zero_steps 1 \
    # --use_cfg_zero_star \
    # --use_zero_init \
    # --zero_steps 1 \
    
INPUT_CSV="playground/gigaworld_t2v_prompts.csv"
BASE_OUTPUT_DIR="playground/results"
PLAYGROUND_DIR="playground"

SCORE_TYPE="rating"  # ["raw", "normalized", "rating"]

NUM_WORKERS=32
API_KEY=""
BASE_URL=""

GPU_ID=0

for MODEL_DIR in "$PLAYGROUND_DIR"/*/ ; do
    MODEL_NAME=$(basename "$MODEL_DIR")
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME"

    if [ ! -d "$MODEL_DIR" ]; then
        continue
    fi
    
    echo "Processing model: $MODEL_NAME"
    VIDEO_DIR="$MODEL_DIR"

    # Aesthetic
    CUDA_VISIBLE_DEVICES=$GPU_ID python 0_get_aesthetic.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --clip_model_path "checkpoints/aesthetic_model/ViT-L-14.pt" \
        --aesthetic_model_path "checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth" &

    # Motion Amplitude
    CUDA_VISIBLE_DEVICES=$GPU_ID python 1_get_motion_amplitude.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --num_workers $NUM_WORKERS &

    # Motion Smoothness
    CUDA_VISIBLE_DEVICES=$GPU_ID python 2_get_motion_smoothness.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --smoothness_model_path "checkpoints/amt_model/amt-s.pth" &

    # Semantic
    CUDA_VISIBLE_DEVICES=$GPU_ID python 3_get_semantic.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --semantic_model_path "checkpoints/ViCLIP" &

    # Naturalness
    CUDA_VISIBLE_DEVICES=$GPU_ID python 4_get_naturalness.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --api_key $API_KEY \
        --base_url $BASE_URL \
        --num_workers $NUM_WORKERS &

    # Drifting Aesthetic
    CUDA_VISIBLE_DEVICES=$GPU_ID python 5_get_drifting_aesthetic.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --clip_model_path "checkpoints/aesthetic_model/ViT-L-14.pt" \
        --aesthetic_model_path "checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth" &

    # Drifting Motion Smoothness
    CUDA_VISIBLE_DEVICES=$GPU_ID python 6_get_drifting_motion_smoothness.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --smoothness_model_path "checkpoints/amt_model/amt-s.pth" &

    # Drifting Semantic
    CUDA_VISIBLE_DEVICES=$GPU_ID python 7_get_drifting_semantic.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --semantic_model_path "checkpoints/ViCLIP" &

    # Drifting Naturalness
    CUDA_VISIBLE_DEVICES=$GPU_ID python 8_get_drifting_naturalness.py \
        --input_csv $INPUT_CSV \
        --video_dir $VIDEO_DIR \
        --output_path $OUTPUT_DIR \
        --api_key $API_KEY \
        --base_url $BASE_URL \
        --num_workers $NUM_WORKERS &

    wait

    # Merge All Scores
    python 9_merge_all_scores.py \
        --input_dir "$OUTPUT_DIR" \
        --is_long
done

# Merge All Results
python 10_merge_all_results.py \
    --input_dir "$BASE_OUTPUT_DIR" \
    --score_type "$SCORE_TYPE"
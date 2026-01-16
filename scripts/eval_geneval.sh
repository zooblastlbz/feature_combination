# Define variables for the script paths and directories
CUDA_VISIBLE_DEVICES=5
SCRIPT_EVALUATE="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/text_encoder/geneval/evaluation/evaluate_images.py"
SCRIPT_SUMMARY="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/text_encoder/geneval/evaluation/summary_scores.py"
MODEL_EVAL_DIR="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/text_encoder/geneval/evaluation/models/"

# Define variables for the specific experiment run
OUTPUT_BASE_DIR="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/feature_combination/output"
EXPERIMENT_NAME="256-baseline-norm-4b"
CHECKPOINT_STEP="500000"
SCALES="10"  # 与生成时的 scale 保持一致，决定子目录名
STEPS="50"   # 与生成时的步数保持一致，决定子目录名

# Construct the full paths using the variables
CHECKPOINT_DIR="$OUTPUT_BASE_DIR/$EXPERIMENT_NAME/checkpoint-$CHECKPOINT_STEP"
INPUT_GENEVAL_DIR="$CHECKPOINT_DIR/geneval-$SCALES-$STEPS"
OUTPUT_JSON_FILE="$CHECKPOINT_DIR/geneval-$SCALES-$STEPS-eval.json"
SCORE_PATH="$CHECKPOINT_DIR/geneval-$SCALES-$STEPS-eval-summary.json"
# --- Execute the commands ---


# Command 1: Evaluate images
#CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python "$SCRIPT_EVALUATE" \
  "$INPUT_GENEVAL_DIR" \
  --outfile "$OUTPUT_JSON_FILE" \
  --model-path "$MODEL_EVAL_DIR"

# Command 2: Summarize scores
python "$SCRIPT_SUMMARY" \
  "$OUTPUT_JSON_FILE" --save_path "$SCORE_PATH" 
#!/bin/bash

# 单 checkpoint 的 GenAI-Bench 评估脚本，风格与 eval_geneval.sh 一致

# === 脚本路径与模型配置 ===
EVAL_SCRIPT="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/text_encoder/t2v_metrics/genai_bench/custom_evaluate_local.py"
GPT_MODEL="Qwen3-VL-235B-A22B-Instruct"

# === 实验与检查点配置 ===
OUTPUT_BASE_DIR="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/feature_combination/output"
EXPERIMENT_NAME="256-AdaFuseDiT-timewise-LNzero"
CHECKPOINT_STEP="500000"
GENAIBENCH_SCALE="6"   # 与生成时的 scale 保持一致，决定子目录名

IMAGE_DIR=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/checkpoint-${CHECKPOINT_STEP}/genaibench-${GENAIBENCH_SCALE}
# === 评估输出配置 ===
RESULT_DIR=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/checkpoint-${CHECKPOINT_STEP}/genaibench-${GENAIBENCH_SCALE}-eval
NUM_THREADS=20
NUM_IMAGES_PER_PROMPT=1

# === 构造路径 ===
GEN_MODEL=${EXPERIMENT_NAME}-${CHECKPOINT_STEP}
# 确保结果目录存在
mkdir -p "$RESULT_DIR"

echo "评估目录: $GEN_MODEL"

python "$EVAL_SCRIPT" \
  --output_dir "$IMAGE_DIR" \
  --gen_model "$GEN_MODEL" \
  --model "$GPT_MODEL" \
  --num_threads "$NUM_THREADS" \
  --result_dir "$RESULT_DIR" \
  --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT"

echo "评估完成。"

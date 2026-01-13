#!/bin/bash
set -euo pipefail

# 单 checkpoint 的 DrawBench 评估脚本，结构对齐 eval_genaibench.sh
# 可通过环境变量覆盖，或直接编辑下方常量。

# === 脚本路径与模型配置 ===
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="${ROOT_DIR}/evaluation/eval_drawbench_unifiedreward2.py"
SERVER_URL="${SERVER_URL:-http://localhost:8080}"

# === 实验与检查点配置 ===（仿 eval_genaibench 固定写法，可手动改）
OUTPUT_BASE_DIR="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/feature_combination/output"
EXPERIMENT_NAME="256-AdaFuseDiT-timewise-LNzero-4b"
CHECKPOINT_STEP="500000"
GEN_SCALE="7"     # 与生成时的 guidance scale 对应子目录
GEN_STEPS="28"    # 与生成步数对应子目录
METADATA_FILE="${ROOT_DIR}/configs/drawbench/metadata.json"
BATCH_SIZE=8

# === 构造路径 ===
IMAGES_DIR=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/checkpoint-${CHECKPOINT_STEP}/draw-${GEN_SCALE}-${GEN_STEPS}
RESULT_FILE=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/checkpoint-${CHECKPOINT_STEP}/drawbench-${GEN_SCALE}-${GEN_STEPS}-eval.jsonl

# 确保结果目录存在
mkdir -p "$(dirname "$RESULT_FILE")"

echo "评估目录      : $IMAGES_DIR"
echo "元数据       : $METADATA_FILE"
echo "评测脚本     : $EVAL_SCRIPT"
echo "服务地址     : $SERVER_URL"
echo "结果文件     : $RESULT_FILE"
echo "Batch Size   : $BATCH_SIZE"

python "$EVAL_SCRIPT" \
  --images-dir "$IMAGES_DIR" \
  --metadata-file "$METADATA_FILE" \
  --server-url "$SERVER_URL" \
  --output-file "$RESULT_FILE" \
  --batch-size "$BATCH_SIZE"

echo "评估完成，结果已写入：$RESULT_FILE"

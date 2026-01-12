#!/bin/bash
set -euo pipefail

# 评价使用 unifiedreward2 的 DrawBench 生成结果。
# 用法（可通过环境变量覆盖）：
#   IMAGES_DIR=/path/to/draw-7-28 \
#   METADATA_FILE=/path/to/metadata.json \
#   SERVER_URL=http://localhost:8080 \
#   OUTPUT_FILE=/path/to/results.jsonl \
#   BATCH_SIZE=8 \
#   bash scripts/eval_drawbench.sh
#
# 也可以直接传入前两个参数：
#   bash scripts/eval_drawbench.sh /path/to/draw-7-28 /path/to/metadata.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="${ROOT_DIR}/evaluation/eval_drawbench_unifiedreward2.py"

# 默认值可通过环境变量覆盖
DEFAULT_IMAGES_DIR="${ROOT_DIR}/output/draw-7-28"
DEFAULT_METADATA_FILE="${ROOT_DIR}/configs/drawbench/metadata.json"
IMAGES_DIR="${1:-${IMAGES_DIR:-$DEFAULT_IMAGES_DIR}}"
METADATA_FILE="${2:-${METADATA_FILE:-$DEFAULT_METADATA_FILE}}"
SERVER_URL="${SERVER_URL:-http://localhost:8080}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT_DIR}/output/drawbench_unifiedreward2.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "EVAL_SCRIPT   : $EVAL_SCRIPT"
echo "IMAGES_DIR    : $IMAGES_DIR"
echo "METADATA_FILE : $METADATA_FILE"
echo "SERVER_URL    : $SERVER_URL"
echo "OUTPUT_FILE   : $OUTPUT_FILE"
echo "BATCH_SIZE    : $BATCH_SIZE"
echo

python "$EVAL_SCRIPT" \
  --images-dir "$IMAGES_DIR" \
  --metadata-file "$METADATA_FILE" \
  --server-url "$SERVER_URL" \
  --output-file "$OUTPUT_FILE" \
  --batch-size "$BATCH_SIZE"

echo "评估完成，结果已写入：$OUTPUT_FILE"

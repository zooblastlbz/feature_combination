python utils/save_pipeline.py \
    --checkpoint /ytech_m2v8_hdd/workspace/kling_mm/libozhou/feature_combination/output/256-AdaFuseDiT-timewise-LNzero/checkpoint-370000 \
    --type adafusedit \
    --vae /ytech_m2v5_hdd/workspace/kling_mm/Models/Lumina-Image-2.0/vae \
    --scheduler /ytech_m2v5_hdd/workspace/kling_mm/Models/Lumina-Image-2.0/scheduler/ \
    --llm_path /ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen3-VL-4B-Instruct/ \
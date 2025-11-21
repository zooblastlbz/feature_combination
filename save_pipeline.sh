python utils/save_pipeline.py \
    --checkpoint /ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/output/256-AdaFuseDiT-timewise/30000 \
    --type adafusedit \
    --trainer deepspeed \
    --vae /ytech_m2v5_hdd/workspace/kling_mm/Models/Lumina-Image-2.0/vae \
    --scheduler /ytech_m2v5_hdd/workspace/kling_mm/Models/Lumina-Image-2.0/scheduler/ \
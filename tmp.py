from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3Model.from_pretrained("/ytech_m2v5_hdd/workspace/kling_mm/Models/sam3/").to(device)
processor = Sam3Processor.from_pretrained("/ytech_m2v5_hdd/workspace/kling_mm/Models/sam3/")

# Load image
image_url = "/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/mini_dataset_100/images/2.JPEG"
image = Image.open(image_url).convert("RGB")

# Segment using text prompt
inputs = processor(images=image, text="the man in white jack", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Post-process results
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} objects")
print("boxes:", results["boxes"])
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores

import json
import os
from pydoc import text
base_dir='/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/UMA/data/imagenet1k/ImageNet-1K/'
with open('/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/UMA/data/imagenet1k/ImageNet-1K/sharegpt_vqa_t2i.json', 'r') as f:
    data = json.load(f)
    
image_text_pair=[]
num=500000
for item in data:
    image=os.path.join(base_dir,item['image'])
    text=item['conversations'][-1]['value']
    image_text_pair.append({'image':image,'text':text})
    if len(image_text_pair)>=num:
        break
    

with open('/ytech_m2v5_hdd/workspace/kling_mm/libozhou/feature_combination/data/imagenet1k_512_sharegpt_vqa_t2i.json', 'w') as f:
    json.dump(image_text_pair, f, indent=4,ensure_ascii=False)
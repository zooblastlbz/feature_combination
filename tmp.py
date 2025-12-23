



import json
data_path="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/text_encoder/geneval/prompts/evaluation_metadata.jsonl"
data = []
with open(data_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        item = json.loads(line)
        data.append(item)
for item in data:
    prompt=item['prompt']
    prompt=prompt.replace("1. Overall description: ","").replace("2. Main object description: ","").replace("3. Background description: ","").replace("4. Movement description: ","").replace("5. Style description: ","").replace("6. Camera description: ","")
    item['prompt']=prompt
with open("/ytech_m2v5_hdd/workspace/kling_mm/libozhou/text_encoder/geneval/prompts/evaluation_metadata_cleaned.jsonl", 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')

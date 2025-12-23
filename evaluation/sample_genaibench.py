'''
{
    "00000": {
        "id": "00000",
        "prompt": "A male baker in a traditional bakery is using a long peel to pull loaves of bread out of a wood-fired oven, surrounded by shelves of freshly baked bread and various baking tools.A single adult male baker of Caucasian ethnicity is positioned in the center of the scene. He has a medium build and is wearing a white chef's hat, a white short-sleeved shirt, and a white apron. He is standing in front of a large brick wood-fired oven, holding a long wooden peel with both hands, and appears to be pulling a loaf of freshly baked bread out of the oven. The baker is surrounded by various types of bread, including round loaves and baguettes, displayed on wooden trays and shelves in the foreground and background. The bread is golden brown and appears freshly baked.The background features a traditional bakery setting with exposed brick walls and a large, rustic wood-fired oven. There are several copper pots and pans hanging on the wall above the oven. Shelves filled with loaves of bread and baking supplies are visible, as well as a large window letting in natural daylight, creating a warm and inviting atmosphere. The lighting is soft and natural, with sunlight streaming through the window, casting gentle shadows. The overall scene is cozy and filled with the aroma and warmth of freshly baked bread.A male baker wearing a white chef's hat and apron is holding a long wooden peel with both hands, pulling a loaf of freshly baked bread out of a wood-fired oven. The oven is open, with flames visible inside, and several loaves of bread are already inside the oven. The baker is focused on the task, with a neutral expression.The scene is realistic. Warm, rustic ambiance with soft lighting, highlighting artisanal bread and traditional baking tools in a cozy, vintage kitchen setting.The scene is captured using a standard lens, providing a balanced perspective suitable for the indoor setting. The scene type is a medium shot, focusing on the baker and the oven, with the surrounding bakery environment visible. The angle of view is horizontal, capturing the scene from the side, which allows for a clear view of the baker's actions and the oven. The main subject, the baker, is positioned slightly to the right of the center in the frame. Clear background.",
        "prompt in Chinese": "一位面包师正从面包店的烤箱中取出刚烤好的面包。",
        "models": {
            "DALLE_3": [
                5,
                5,
                5
            ],
            "SDXL_Turbo": [
                2,
                4,
                3
            ],
            "DeepFloyd_I_XL_v1": [
                4,
                3,
                3
            ],
            "Midjourney_6": [
                3,
                3,
                3
            ],
            "SDXL_2_1": [
                3,
                2,
                2
            ],
            "SDXL_Base": [
                3,
                3,
                2
            ]
        },
        "short_prompt": "A baker pulling freshly baked bread out of an oven in a bakery."
    },
}
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from accelerate import PartialState
import fire
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import torch
import json
from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP, AdaFuseDiTPipeline


def load_pipeline(model_type: str, ckpt_path: str, dtype):
    if model_type == "dit":
        pipeline = DiTPipeline.from_pretrained(ckpt_path, torch_dtype=dtype).to("cuda")
    elif model_type == "fuse-dit":
        pipeline = FuseDiTPipeline.from_pretrained(ckpt_path, torch_dtype=dtype).to("cuda")
    elif model_type == "fuse-dit-clip":
        pipeline = FuseDiTPipelineWithCLIP.from_pretrained(ckpt_path, torch_dtype=dtype).to("cuda")
    elif model_type == "adafusedit":
        pipeline = AdaFuseDiTPipeline.from_pretrained(ckpt_path, torch_dtype=dtype).to("cuda")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@torch.no_grad()
def generate(opt):
    
    with open(opt.gen.metadata_file, "r") as f:
        metadata = json.load(f)
    prompts=[]
    for item in metadata.items():
        prompts.append(item)
    
    
    dtype_str = opt.pipeline.get("dtype", "bf16")
    if dtype_str == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_str == "fp16":
        torch_dtype = torch.float16
    elif dtype_str == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    distributed_state = PartialState()
    with distributed_state.split_between_processes(prompts) as samples:
        for model in tqdm(opt.pipeline.ckpt_path):
            os.makedirs(os.path.join(model, f"genaibench-{int(opt.gen.scale)}"), exist_ok=True)
            pipe = load_pipeline(opt.pipeline.model_type, os.path.join(model, "pipeline"), torch_dtype)
            for sample in tqdm(samples):
                prompt = open(os.path.join(opt.gen.prompts_dir, sample)).read()

                prompt = sample['prompt']
                generator = torch.manual_seed(opt.gen.seed)
                with torch.autocast("cuda", dtype=torch_dtype):
                    images = pipe(
                        prompt=opt.gen.instruction + prompt,
                        height=opt.gen.H,
                        width=opt.gen.W,
                        num_inference_steps=opt.gen.steps,
                        guidance_scale=opt.gen.scale,
                        num_images_per_prompt=1,
                        negative_prompt=opt.gen.negative_prompt or None,
                        generator=generator,
                        instruction=opt.gen.instruction,
                    )[0]
                
                image=images[0]
                id=sample[0]
                image.save(os.path.join(model, f"genaibench-{opt.gen.scale}", f"{id}.png"))


def main(config_file):
    hparams = OmegaConf.load(config_file)
    generate(hparams)


if __name__ == "__main__":
    fire.Fire(main)
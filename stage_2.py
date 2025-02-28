from diffusers import DiffusionPipeline, AutoPipelineForImage2Image, AutoPipelineForInpainting
from diffusers.utils import make_image_grid, load_image
import torch
import random
import argparse

parser = argparse.ArgumentParser(description='Stage-2')
parser.add_argument('-s', '--seed', help='seed, random if not specified', required=False)
parser.add_argument('-i', '--init_image', help='init image', default='./grabcut-image.jpg', required=False)
parser.add_argument('-m', '--mask_image', help='mask image', default='./grabcut-smask.jpg', required=False)
parser.add_argument('-o', '--output', help='output image', default='./s2.jpg', required=False)
args = parser.parse_args()

#初始化，加载所需的LoRA权重
basic_pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
# load LoRA weight
basic_pipe.load_lora_weights("./models/lora-trained-xl-fu/pytorch_lora_weights.safetensors", adapter_name="cn_fu_pca")
basic_pipe.load_lora_weights("./models/lora-trained-xl-flower/pytorch_lora_weights.safetensors", adapter_name="flower_pca")
basic_pipe.load_lora_weights("./models/lora-trained-xl-fish/pytorch_lora_weights.safetensors", adapter_name="fish_pca")
basic_pipe.load_lora_weights("./models/lora-trained-xl-loong/pytorch_lora_weights.safetensors", adapter_name="loong_pca")
basic_pipe.load_lora_weights("./models/lora-trained-xl-bat/pytorch_lora_weights.safetensors", adapter_name="bat_pca")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
refiner.to("cuda")

if args.seed:
    seed = int(args.seed)
else:
    seed = int(random.random()*1000000000)
print(f"seed = {seed}")

flower_prompt = "(<cn-fu-pca>), full filled with (<flower-pca>) patterns, fine textures, Delicate petal details, Lush botanical patterns, solid bright red, white background, Smooth shapes, intricate and elegant, hopeful, highly detailed, ultra-clear. "
fish_prompt = "(<cn-fu-pca>), full of (<fish-pca>) patterns, fine textures, intricate fish scale texture, two Entire fish motif clearly seperated by (distinct boundaries), solid bright red, white background, (full-filled areas), red contours, Smooth shapes, intricate and elegant, hopeful, highly detailed, ultra-clear. "
loong_prompt = "(<cn-fu-pca>), (<loong-pca>) patterns, one entire chinese loong, fine textures, intricate scale texture, solid bright red, white background, (full-filled areas), red contours, Smooth shapes, intricate and elegant, hopeful, highly detailed, ultra-clear. "
bat_prompt = "(<cn-fu-pca>), (<bat-pca>) patterns, one entire bat, fine textures, solid bright red, white background, (full-filled areas), red contours, Smooth shapes, intricate and elegant, hopeful, highly detailed, ultra-clear. "
fishandflower_prompt = "(<cn-fu-pca>), full of (<fish-pca>) patterns or (<flower-pca>) patterns,big proportion of (<flower-pca>) patterns,fine textures, solid bright red, one entire fish seperated with flowers by (distinct boundaries) ,(full-filled areas), white background, Smooth shapes, intricate and elegant, hopeful, highly detailed, ultra-clear, fine textures."
neg_prompt = "Unfilled areas, Chinese script, Blurry textures, Low resolution, Low detail"
refine_prompt = "chinese paper cut art, ((Vermilion)), (closed border), white background"

def inpaint_image(init_image, mask_image, prompt, neg_prompt, refine_prompt, adapters, adapter_weights, n_steps=25):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    basic_pipe.set_adapters(adapters, adapter_weights=adapter_weights)
    basic_pipe.to("cuda")
    inpaint_pipe = AutoPipelineForInpainting.from_pipe(basic_pipe).to("cuda")
    latent_image = inpaint_pipe(prompt=prompt, 
                                image=init_image, mask_image=mask_image, 
                                strength=1.0, guidance_scale=12.5, 
                                num_inference_steps=n_steps, output_type="latent").images[0]
    image = refiner(prompt=refine_prompt, num_inference_steps=n_steps, generator=generator, image=latent_image).images[0]
    return image

init_image, mask_image = load_image(args.init_image), load_image(args.mask_image)
flower_image = inpaint_image(init_image, mask_image, flower_prompt, neg_prompt, refine_prompt, 
                            ["cn_fu_pca", "flower_pca"], [1.0, 0.6])
fishandflower_image = inpaint_image(init_image, mask_image, fishandflower_prompt, neg_prompt, refine_prompt, 
                            ["cn_fu_pca", "flower_pca", "fish_pca"], [1.0, 0.6, 0.6])
fish_image = inpaint_image(init_image, mask_image, fish_prompt, neg_prompt, refine_prompt, 
                            ["cn_fu_pca", "fish_pca"], [1.0, 0.6])
bat_image = inpaint_image(init_image, mask_image, bat_prompt, neg_prompt, refine_prompt, 
                            ["cn_fu_pca", "bat_pca"], [1.0, 0.6])
res = make_image_grid([init_image, fish_image, fishandflower_image], rows=1, cols=3)
#res = make_image_grid([init_image, flower_image, bat_image], rows=1, cols=3)
res.save(args.output)


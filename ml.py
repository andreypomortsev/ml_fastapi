import os
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

load_dotenv()
TOKEN = os.getenv('TOKEN')
device = "CUDA"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=TOKEN
)

pipe = pipe.to(device)

prompt = input()

image = pipe(prompt).images[0]  
    
image.save(f"{prompt}.png")
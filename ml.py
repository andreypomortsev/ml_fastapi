import os
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image

load_dotenv()
TOKEN = os.getenv('TOKEN')

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16, 
    revision="fp16",
    use_auth_token=TOKEN
)

pipe = pipe.to('cuda')

prompt = input('WHa ')

image = pipe(prompt).images[0]  
    
image.save(f"{prompt}.png")
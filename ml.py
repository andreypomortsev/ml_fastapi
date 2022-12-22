import os
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline
# from PIL.Image import Image
import gc

def ram_cleaning() -> None:
    gc.collect()
    torch.cuda.empty_cache()

load_dotenv()
TOKEN = os.getenv('TOKEN')

ram_cleaning()

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype = torch.float16,
    revision = "fp16",
    use_auth_token = TOKEN
)

pipe = pipe.to('cuda')

prompt = "A handsome bunny abasks on the beach"
image = pipe(f"{prompt}").images[0]  
image.save(f"{prompt}.png")

ram_cleaning()
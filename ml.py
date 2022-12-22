import os
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline
# from PIL.Image import Image
import gc

gc.collect()
load_dotenv()
TOKEN = os.getenv('TOKEN')

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype = torch.float16,
    revision = "fp16",
    use_auth_token = TOKEN
)
torch.cuda.empty_cache()
pipe.batch_size = 1
pipe = pipe.to('cuda')

prompt = "A handsome bunny abasks on the beach"

image = pipe(f"{prompt}").images[0]  
    
image.save(f"{prompt}.png")
torch.cuda.empty_cache()
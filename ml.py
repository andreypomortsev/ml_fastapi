import os
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL.Image import Image

load_dotenv()
TOKEN = os.getenv('TOKEN')
model_id = "CompVis/stable-diffusion-v1-4"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler = scheduler,
    torch_dtype = torch.float16,
    use_auth_token = TOKEN
)

pipe = pipe.to('cuda') #need to rewrite for diff. types of devices cpu gpu tpu

prompt = "A handsome bunny abasks on the beach"

def obtain_image(
    prompt: str,
    *,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> Image:
    generator = None if seed is None else torch.Generator('cuda')
    print(f"Using device: {pipe.device}")
    image: Image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]
    return image

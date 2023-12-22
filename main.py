import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from ml import obtain_image

app = FastAPI()

COMMAND = "pip install torch==1.12.1 torchvision==0.13.1 \
--extra-index-url https://download.pytorch.org/whl/cu113"
torch_command = os.environ.get("TORCH_COMMAND", COMMAND)
commandline_args = os.environ.get("COMMANDLINE_ARGS", "--skip-torch-cuda-test")


@app.get("/generate")
def generate_image(prompt: str, steps: int):
    """
    Endpoint that generates an image with given prompt and number of inference steps.
    Returns the generated image in a FileResponse.
    """
    image = obtain_image(prompt, num_inference_steps=steps)
    image.save(f"{prompt}.png")
    return FileResponse(f"{prompt}.png")

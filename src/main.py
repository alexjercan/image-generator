from fastapi import FastAPI, Response
from diffusers import DiffusionPipeline
import torch
from io import BytesIO
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(
        alias="prompt", descrption="The text prompt to generate an image from."
    )

    model_config = {
        "json_schema_extra": {
            "example": {"prompt": "An astronaut riding a green horse"}
        }
    }


app = FastAPI()

# Load the model once at startup
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    low_cpu_mem_usage=True,
    variant="fp16",
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()


@app.post("/generate")
async def generate_image(request: GenerateRequest):
    # Generate the image
    image = pipe(request.prompt, num_inference_steps=20).images[0]

    # Save the image to a BytesIO object
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # Return the image as a response
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

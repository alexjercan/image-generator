from PIL import Image
from fastapi import FastAPI, Response, UploadFile
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
from io import BytesIO


class PipeCache:
    def __init__(self):
        self.pipes = {}

    def get_pipe(self, pipe_name: str):
        if pipe_name not in self.pipes:
            self.load_pipe(pipe_name)

        return self.pipes[pipe_name]

    def load_pipe(self, pipe_name: str):
        if pipe_name == "text2image":
            pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                variant="fp16",
            )
        elif pipe_name == "image2image":
            pipe = AutoPipelineForImage2Image.from_pipe(self.get_pipe("text2image"))
        else:
            raise ValueError(f"Unknown pipeline: {pipe_name}")

        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        self.pipes[pipe_name] = pipe


app = FastAPI()
cache = PipeCache()


POSITIVE_KEYWORDS = "4k,ultrarealistic,sharp focus,highly detailed,cinematic lighting\
,octane render,global illumination,bokeh,photorealistic,dramatic lighting\
,award-winning photograph,studio lighting,dslr,hyperrealism,volumetric lighting\
,beautiful composition,rule of thirds,epic scene,professional photograph,symmetry\
,realistic skin texture,vibrant colors,fine details,masterpiece,perfect anatomy\
,natural lighting,film grain,artstation,trending on artstation,unreal engine,hdr\
,fashion editorial,high quality,glossy finish,beautiful eyes,symmetrical face\
,full body shot,in focus"
NEGATIVE_KEYWORDS = "blurry,low quality,poorly drawn,ugly,disfigured,amateurish\
,low resolution,grainy,oversaturated,underexposed,poor lighting,unrealistic proportions\
,cluttered background,bad composition,unprofessional,unappealing colors,lack of detail\
,unfocused,awkward pose"


@app.post("/api/generate/text2image")
async def generate_text_to_image(prompt: str):
    text2image_pipe = cache.get_pipe("text2image")

    positive_prompt = prompt + " " + POSITIVE_KEYWORDS
    negative_prompt = NEGATIVE_KEYWORDS

    image = text2image_pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=1024,
        width=1024,
        num_images_per_prompt=1,
    ).images[0]

    # Save the image to a BytesIO object
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # Return the image as a response
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")


@app.post("/api/generate/image2image")
async def generate_image_to_image(prompt: str, image: UploadFile):
    image2image_pipe = cache.get_pipe("image2image")

    positive_prompt = prompt + " " + POSITIVE_KEYWORDS
    negative_prompt = NEGATIVE_KEYWORDS

    image_bytes = await image.read()
    input_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    generated_image = image2image_pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=0.5,
        height=input_image.height,
        width=input_image.width,
        num_images_per_prompt=1,
        image=input_image,
    ).images[0]

    # Save the generated image to a BytesIO object
    img_byte_arr = BytesIO()
    generated_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # Return the generated image as a response
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

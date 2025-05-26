# Autors: Sergejs Kiseļovs
# ControlNet parametru kombināciju ģenerēšana
# Nepieciešamo bibliotēku instalēšana
# pip install diffusers transformers accelerate safetensors torch opencv-python --quiet

import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image
from io import BytesIO
from itertools import product
import os


# ControlNet modeļa ielāde (Canny malu noteikšana)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
).to("cuda")

# Galvenā Stable Diffusion modeļa ielāde
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Funkcija lietotāja attēla ielādei
def load_user_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        img = Image.open(BytesIO(uploaded[filename])).convert("RGB")
        return img
    return None

# --- Iestatījumi --- #
image_paths = [
    "/content/img.jpg",
    "/content/img1.jpg",
    "/content/img2.jpg",
    "/content/img3.jpg",
    "/content/img4.jpg",
    "/content/img5.jpg",
    "/content/img6.jpg"
]

prompts = [
    "style of anime",
    "style of simpsons",
    "style of disney cartoons",
    "style of pixar",
    "style of Makoto Shinkai"
]

seeds = [0, 42, 1234, 999, 9999, 357646726838, 357646726838, 78097968758234, 1259987876656]  # dažādi seed vērtības

output_dir = "/content/Style_transfer/control_net"

controlnets = {
    "Canny": "lllyasviel/sd-controlnet-canny",
    "Dziļums": "lllyasviel/sd-controlnet-depth",
    "Segmentācija": "lllyasviel/sd-controlnet-seg"
}

def apply_canny(image):
    image = np.array(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_gray, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

def load_controlnet_model(model_name):
    return ControlNetModel.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

def generate_image(controlnet_model, image, prompt, seed):
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet_model,
        torch_dtype=torch.float16
    ).to("cuda")
    generator = torch.manual_seed(seed)
    return pipe(prompt=prompt, image=image, num_inference_steps=100, guidance_scale=10, generator=generator).images[0]

# --- Visu kombināciju ģenerēšana --- #
for img_path, prompt, seed, (name, model_path) in product(image_paths, prompts, seeds, controlnets.items()):
    print(f"\nApstrāde: Modelis={name}, Attēls={img_path}, Apraksts={prompt}, Sēkla={seed}")

    # Attēla ielāde un sagatavošana
    image = Image.open(img_path).convert("RGB")
    image = image.resize((image.width, image.height))  # Pēc vajadzības var mainīt izmēru

    processed_image = apply_canny(image) if name == "Canny" else image
    controlnet = load_controlnet_model(model_path)

    output_image = generate_image(controlnet, processed_image, prompt, seed)

    # Unikāla faila nosaukuma ģenerēšana
    image_base = os.path.splitext(os.path.basename(img_path))[0]
    safe_prompt = prompt.replace(" ", "_")[:50]
    out_filename = f"{image_base}_{name}_steps100_scale10_seed{seed}_{safe_prompt}.jpg"
    out_path = os.path.join(output_dir, out_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    output_image.save(out_path)
    print(f"Saglabāts: {out_path}")

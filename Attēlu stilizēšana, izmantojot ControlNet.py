# Autors: Sergejs Kiseļovs
# Attēlu stilizēšana, izmantojot ControlNet
# Nepieciešamo bibliotēku instalēšana
# pip install diffusers transformers accelerate safetensors torch opencv-python --quiet

import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image
from google.colab import files
from io import BytesIO

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

# Funkcija Canny filtra pielietošanai
def apply_canny(image):
    image = np.array(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_gray, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

# Funkcija ControlNet modeļa ielādei
def load_controlnet_model(model_name):
    return ControlNetModel.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

# Funkcija attēla ģenerēšanai
def generate_image(controlnet_model, image, prompt, seed):
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet_model,
        torch_dtype=torch.float16
    ).to("cuda")
    generator = torch.manual_seed(seed)
    return pipe(prompt=prompt, image=image, num_inference_steps=100, guidance_scale=10, generator=generator).images[0]

# Ielādēt attēlu
image = Image.open("img.jpg").convert("RGB")

# Iegūt izmērus un samazināt attēlu, ja nepieciešams
width, height = image.size
dell = 1    # cik reizes samazināt attēlu, ja nepieciešams
image = image.resize((width // dell, height // dell))

# Promt un seed
prompt = "anime style"
seed = 2025  # Fiksēta sēkla

# Pieejamie ControlNet modeļi
controlnets = {
    "Canny": "lllyasviel/sd-controlnet-canny",
    "Depth": "lllyasviel/sd-controlnet-depth",
    "Segmentation": "lllyasviel/sd-controlnet-seg"
}

# Ģenerēt attēlus ar dažādiem ControlNet modeļiem
for name, model_path in controlnets.items():
    print(f"Apstrāde ar {name} modeli...")
    controlnet = load_controlnet_model(model_path)
    processed_image = apply_canny(image) if name == "Canny" else image
    output_image = generate_image(controlnet, processed_image, prompt, seed)
    output_image.save(f"{name} control_net steps100 scale10 d{dell} seed:{seed} {prompt}.jpg")
    print(f"Saglabāts: {name} control_net steps100 scale10 d{dell} seed:{seed} {prompt}.jpg")

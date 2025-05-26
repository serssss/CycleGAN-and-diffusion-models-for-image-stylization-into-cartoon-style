# Autors: Sergejs Kiseļovs
# Attēlu stilizēšana ar Img2img, stilizācijas parametru iterācija

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import random

# Mapes
input_dir = "photo_dir"   # Mape ar attēliem
output_base_dir = "output_dir"

# Parametri
prompts = [
    "anime stils",
    "simpsoni stils",
    "disneja multfilmu stils",
    "pixar stils"
]  # Dažādi uzvednes

seeds = [42, 1234, 900, 2025, 7245685797642454, 357646726838, 357646726838]  # Dažādas sēklas

num_inference_steps = 100
noise_strength_values = [0.4, 0.5, 0.6]
guidance_scale_values = [10.0, 15.0, 20.0]

# Modeļa ielāde
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Attēlu saraksts
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
print(f"Atrasti {len(image_files)} attēli.")

# Apstrāde
for img_idx, img_name in enumerate(image_files, 1):
    print(f"\n[{img_idx}/{len(image_files)}] Apstrādājas: {img_name}")
    image_path = os.path.join(input_dir, img_name)
    base_name = os.path.splitext(img_name)[0]

    # Ielāde un izmēra korekcija
    init_image = Image.open(image_path).convert("RGB")
    width, height = init_image.size
    if width % 8 != 0 or height % 8 != 0:
        width -= width % 8
        height -= height % 8
        init_image = init_image.resize((width, height))

    # Rezultātu mapes izveide
    output_dir = os.path.join(output_base_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)

    total = len(prompts) * len(seeds) * len(noise_strength_values) * len(guidance_scale_values)
    count = 1

    # Ģenerēšana visām kombinācijām
    for prompt in prompts:
        for seed in seeds:
            generator = torch.Generator(device="cuda").manual_seed(seed)

            for strength in noise_strength_values:
                for guidance_scale in guidance_scale_values:
                    print(f"  [{count}/{total}] uzvedne='{prompt}', sēkla={seed}, stiprums={strength}, vadības_mērogs={guidance_scale}")

                    output = pipe(
                        prompt=prompt,
                        image=init_image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        num_inference_steps=num_inference_steps
                    )

                    result_image = output.images[0]

                    # Drošs faila nosaukums
                    safe_prompt = prompt.replace(" ", "_").replace(",", "")
                    filename = (
                        f"{base_name}_uzvedne-{safe_prompt}_seed{seed}_strength{strength}_scale{guidance_scale}.png"
                    )
                    result_image.save(os.path.join(output_dir, filename))

                    count += 1

print("\n Visi attēli un kombinācijas veiksmīgi apstrādāti.")


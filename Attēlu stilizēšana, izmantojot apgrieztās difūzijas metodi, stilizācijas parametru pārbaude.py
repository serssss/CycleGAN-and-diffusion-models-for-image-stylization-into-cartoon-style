# Autors: Sergejs Kiseļovs
# Attēlu stilizēšana, izmantojot apgrieztās difūzijas metodi, stilizācijas parametru pārbaude

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# --- Parametri ---
input_dir = "photo_dir"   # Mape ar oriģinālajiem attēliem
output_base_dir = "output_dir"

prompts = [
    "amine style",
    "simpsons style",
    "disney cartoons style",
    "pixar style"
]  # Dažādi uzvednes (promti)

seeds = [42, 1234, 900, 2025, 7245685797642454, 357646726838, 357646726838]  # Dažādi sēklas skaitļi

num_inference_steps = 100

noise_strength_values = [0.4, 0.5, 0.6]
guidance_scale_values = [10.0, 15.0, 20.0]

# --- Modeļa ielāde ---
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# --- Attēlu failu saraksta iegūšana ---
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
print(f"Atrast{i} {len(image_files)} attēl{i} apstrādei.")

for img_idx, img_file in enumerate(image_files, 1):
    print(f"\n[{img_idx}/{len(image_files)}] Apstrādājam: {img_file}")

    # Attēla ielāde un sagatavošana
    image_path = os.path.join(input_dir, img_file)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image = image.resize(((width // 8) * 8, (height // 8) * 8))  # dalāms ar 8

    # Bāzes nosaukums
    base_name = os.path.splitext(img_file)[0]

    total = len(prompts) * len(seeds) * len(noise_strength_values) * len(guidance_scale_values)
    count = 1

    for prompt in prompts:
        safe_prompt = prompt.replace(" ", "_").replace(",", "").replace(":", "")  # Drošs nosaukums failam

        for seed in seeds:
            generator = torch.Generator(device="cuda").manual_seed(seed)

            for noise_strength in noise_strength_values:
                for guidance_scale in guidance_scale_values:
                    print(f"  [{count}/{total}] prompt='{prompt}', seed={seed}, strength={noise_strength}, scale={guidance_scale}")

                    result = pipe(
                        prompt=prompt,
                        image=image,
                        strength=noise_strength,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )

                    output_image = result.images[0]

                    # Izveidojam izvades mapi (ligzdošana: attēla nosaukums → uzvedne)
                    output_dir = os.path.join(output_base_dir, base_name, safe_prompt)
                    os.makedirs(output_dir, exist_ok=True)

                    filename = f"{base_name}_seed{seed}_strength{noise_strength}_scale{guidance_scale}.jpg"
                    output_image.save(os.path.join(output_dir, filename))

                    count += 1

print("\n Visi attēli un kombinācijas veiksmīgi apstrādāti.")

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

def generate_images(prompt, negative_prompt, num_images_per_prompt,
                    num_inference_steps, height, width, seed, guidance_scale):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_model_or_path = "stabilityai/stable-diffusion-2-1-base"

    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_or_path).to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    
    images = pipeline(prompt=prompt, 
                      num_images_per_prompt=num_images_per_prompt,
                      negative_prompt=negative_prompt, 
                      num_inference_steps=num_inference_steps,
                      height=height, 
                      width=width, 
                      generator=generator,
                      guidance_scale=guidance_scale)["images"]

    return images

prompt = "Generate a high-quality image of a clownfish in an aquarium."
negative_prompt = "It should not contain seahorses"
num_images_per_prompt = 2
num_inference_steps = 50
height = 512
width = 512
seed = 1234
guidance_scale = 7.5

images = generate_images(prompt, negative_prompt, num_images_per_prompt,
                         num_inference_steps, height, width, seed, guidance_scale)

total_images = len(images)
fig, axes = plt.subplots(1, total_images, figsize=(total_images * 10, 10))

if total_images == 1:
    axes.imshow(images[0])
    axes.axis("off")
else:
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")

plt.show()
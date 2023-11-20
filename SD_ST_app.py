import streamlit as st
import pandas as pd
import numpy as np

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

#https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline
# https://huggingface.co/docs/diffusers/v0.19.3/api/pipelines/stable_diffusion/stable_diffusion_xl
#https://github.com/ahgsql/StyleSelectorXL/blob/main/sdxl_styles.json
# https://github.com/huggingface/diffusers/issues/3117 ## Seed, callback function
# https://github.com/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb Experiments with seed

@st.cache_resource
def load_base_model():
    base_pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    base_pipeline = base_pipeline.to("mps")

    return base_pipeline

@st.cache_resource
def load_refiner_model():
    refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", 
                                                                        # text_encoder_2=base_pipeline.text_encoder_2,
                                                                        # vae=base_pipeline.vae,
                                                                        torch_dtype=torch.float16)
    refiner_pipeline = refiner_pipeline.to("mps")

    return refiner_pipeline

base_model_bar = None
num_inference_steps = None
refine_model_bar = None
refine_text_progress = None

def progress(step, timestep, latents):
    global base_model_bar, num_inference_steps
    base_model_bar.progress(int((step+1)*100/num_inference_steps), text="Base Model generation progress ")
    # print(step, timestep, latents[0][0][0][0])

def refiner_progress(step, timestep, latents):
    global refine_model_bar, refine_text_progress
    refine_model_bar.progress(int((step+1)*100/15), text=refine_text_progress)

def main():
    global base_model_bar, num_inference_steps, refine_model_bar
    st.set_page_config(page_title="Generate layouts of homes",
                       page_icon=":home:")
    with st.container():
        prompt = st.text_input("Enter your prompt", value="", max_chars=300)
        negative_prompt = st.text_input("Enter your negative prompt", value="", max_chars=300)
    
    with st.sidebar:
        num_inference_steps = st.sidebar.slider(
                "Inference Steps",
                min_value=1,  # Minimum value
                max_value=100,  # Maximum value
                value=20, # Default value
                step=1  # Step size
            )
        
        num_images_per_prompt = st.sidebar.slider(
                "Number of Images per prompt",
                min_value=1,  # Minimum value
                max_value=8,  # Maximum value
                value=2, # Default value
                step=1  # Step size
            )
        
        guidance_scale = st.sidebar.slider(
                "Guidance scale",
                min_value=1.0,  # Minimum value
                max_value=13.0,  # Maximum value
                value=7.0, # Default value
                step=0.1  # Step size
            )
        
        enable_refiner = st.checkbox("Enable Refiner")
        enable_manual_seed = st.checkbox("Enable Manual Seed")

        if enable_manual_seed:
            seed_input = st.text_input("Manual seed number")

        

    base_pipe = load_base_model()
    if enable_refiner:
        refiner_pipe = load_refiner_model()

    if st.button("Process", key="1"):

        if seed_input.isdigit():
            generator = torch.Generator(device="mps").manual_seed(int(seed_input))
        else:
            st.warning('Seed value is not a integer, proceesing with random seed', icon="⚠️")
            generator = torch.Generator(device="mps")

        st.write("Processing prompt : ", prompt, " with negative prompt : ", negative_prompt)
        base_model_bar = st.progress(0, "Base Model generation progress")

        processed_base_pipe = base_pipe(prompt,
                     negative_prompt=negative_prompt,
                     generator=generator,
                     guidance_scale=guidance_scale,
                     num_images_per_prompt=num_images_per_prompt,
                     num_inference_steps=num_inference_steps,
                     callback=progress,
                     callback_steps=1,
                    #  denoising_end=high_noise_frac,
                    #  prompt_2 = "Van Gogh painting"
                     )
        base_model_bar.empty()

        if enable_refiner:
            for i in range(num_images_per_prompt):
                refine_text_progress = "Refiner Model generation progress Image " + str(i)
                refine_model_bar = st.progress(0, refine_text_progress)
                processed_refined_pipe = refiner_pipe(prompt,
                                                negative_prompt=negative_prompt,
                                                image = processed_base_pipe.images[i],
                                                callback=refiner_progress,
                                                callback_steps=1,)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(processed_base_pipe.images[i], caption="Base "+prompt)

                with col2:
                    st.image(processed_refined_pipe.images[0], caption="Refined "+prompt)
                refine_model_bar.empty()
                
        else:
            for i in range(num_images_per_prompt):
                st.image(processed_base_pipe.images[i], caption="Base "+prompt)


if __name__ == '__main__':
    main()
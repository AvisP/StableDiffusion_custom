# Custom StableDiffusion model
A repo in which customized stable diffusion GUI with simplified use cases are demonstrated

A simple Graphical user interface developed using streamlit that allows the user to generate images using Stable Diffusion XL (SDXL) model using positive and negative prompts. User have the option to enable the refiner model of SDXL as well. Other control parameters available to user are number of inference steps, number of images to generate per prompt and guidance scale. There is also an option to set manual seed for reporoducibility. A progress bar is shown for base as well as refiner model to show the progress during image generation.




## To Do
- [ ] Add the option to allow the user to select various stable diffusion models
- [ ] Add option to allow user to select the Sampling Scheduler and it's parameters
- [ ] Add option to specify a LORA and VAE during image generation
- [ ] Include Control Net so user can perform controlled image generation

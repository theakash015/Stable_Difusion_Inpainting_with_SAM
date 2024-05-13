import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import PIL, cv2
from PIL import Image
from io import BytesIO
from IPython.display import display
import base64, json, requests
from matplotlib import pyplot as plt
import numpy as np
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler #Inpainting pipeline of Stable diffusion
from numpy import asarray
import sys
import os
import gradio as gr
from models import initialize_models
from image_processing import sam_model, on_select, mask_final


def launch_interface():
    mask_generator, pipe = initialize_models()

    with gr.Blocks() as demo:
      gr.Markdown("# Stable Diffusion with Segment Anything!")
      with gr.Row():
        with gr.Column():
            image = gr.Image(type='pil')
        with gr.Column():
            masked_image = gr.Image(type='pil')

      with gr.Row():
        with gr.Column():
            mask=gr.Image(type='pil')

        with gr.Column():
          prompt = gr.Textbox(placeholder="Processing Prompt",label='Prompt')

      with gr.Row():
          button_final = gr.Button("Process Image")

      with gr.Row():
        with gr.Column():
            Original_image=gr.Image(type='pil')
        with gr.Column():
            output = gr.Image(type='pil')

      image.change(sam_model, inputs=image, outputs=masked_image)
      masked_image.select(on_select, inputs=[masked_image,image],outputs=mask)
      button_final.click(mask_final, inputs=[image,mask,prompt],outputs=[Original_image,output])
    
    demo.launch(share=True)
    

if __name__ == "__main__":
    launch_interface()
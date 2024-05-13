import gradio as gr
import numpy as np
import cv2,PIL
from PIL import Image
from matplotlib import pyplot as plt
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
from models import initialize_models

mask_generator, pipe = initialize_models()


def show_anns(anns):
    if len(anns) == 0:
        return
    centroids={}
    # Sort masks by area in descending order
    sorted_anns = sorted(enumerate(anns), key=(lambda x: x[1]['area']), reverse=True)
    ax = plt.gca()

    # Disable autoscale to keep the image size consistent
    ax.set_autoscale_on(False)

    # Iterate through each mask and display it on top of the original image
    for original_idx, ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))

        # Generate a random color for the mask
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]

        # Blend the mask with the image, using 0.35 as the alpha value for transparency
        ax.imshow(np.dstack((img, m*0.35)))

        # Find contours of the mask to compute the centroid
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            M = cv2.moments(cnt)

            # Compute the centroid of the mask if the moment is non-zero
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids[original_idx] = (cx, cy)
                # Plot a marker at the centroid with a star shape
                ax.plot(cx, cy, marker='*', color='white', markersize=10)

    plt.show()

    return centroids

def find_bounding_box(image_array):
    # Find indices of non-white pixels
    non_white_indices = np.where(np.any(image_array[..., :3] != 255, axis=-1))
    # Calculate bounding box coordinates
    min_y, min_x = np.min(non_white_indices, axis=1)
    max_y, max_x = np.max(non_white_indices, axis=1)
    return min_x, min_y, max_x, max_y


def sam_model(img):
    seg = np.asarray(img)
    masks = mask_generator.generate(seg)

    # Display the original image with annotations
    plt.imshow(img)
    c=show_anns(masks)
    plt.axis('off')

    # Render the figure and convert it to a PIL image
    plt_img = plt.gcf()
    plt_img.canvas.draw()
    image_array = np.array(plt_img.canvas.renderer._renderer)

    # Convert RGBA image to RGB
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    pil_image = Image.fromarray(image_array)

    # Resize the image to match the original PIL image size
    pil_image = pil_image.resize(img.size)  # Assuming img is a PIL image

    # Convert the resized PIL image back to a NumPy array
    image_array_resized = np.array(pil_image)

    # Find bounding box of non-white pixels
    min_x, min_y, max_x, max_y = find_bounding_box(image_array_resized)

    # Crop the resized image using the bounding box coordinates
    cropped_image_array = image_array_resized[min_y:max_y, min_x:max_x]

    # Resize the cropped image to match the size of the original image
    cropped_pil_image = Image.fromarray(cropped_image_array)
    cropped_pil_image = cropped_pil_image.resize(img.size)

    return cropped_pil_image

def on_select(masked_image, source_image,evt: gr.SelectData):
    input_points = np.array(evt.index)
    seg = np.asarray(source_image)
    masks = mask_generator.generate(seg)
    c=show_anns(masks)

    min_dist = float('inf')
    nearest_index = None

    for index, coord in c.items():
        dist = np.linalg.norm(np.array(coord) - np.array(input_points))
        if dist < min_dist:
            min_dist = dist
            nearest_index = index

    segmentation_mask=masks[nearest_index]['segmentation']
    stable_diffusion_mask=PIL.Image.fromarray(segmentation_mask)
    return stable_diffusion_mask

def mask_final(source_image, stable_diffusion_mask, inpainting_prompt):
    generator = torch.Generator(device="cuda").manual_seed(155)
    image = pipe(prompt=inpainting_prompt, guidance_scale=30, num_inference_steps=150, generator=generator, image=source_image, mask_image=stable_diffusion_mask).images[0]
    image = image.resize((source_image.width,source_image.height))
    return source_image,image
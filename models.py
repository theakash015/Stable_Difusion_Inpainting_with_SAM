import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler


def initialize_models(model_type='vit_h', device='cuda', sam_checkpoint='sam_vit_h_4b8939.pth'):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.97,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    model_dir = "stabilityai/stable-diffusion-2-inpainting"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir,
                                                          scheduler=scheduler,
                                                          revision="fp16",
                                                          torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    return mask_generator, pipe
#docker run --rm -it --gpus all --entrypoint /bin/bash -v /e/CHIEF_Test/CHIEF:/mnt chiefcontainer/chief:v1.11

import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from models.ctran import ctranspath
from models.CHIEF import CHIEF
import openslide
from PIL import Image, ImageDraw
from tqdm import tqdm
import time


def extract_top_k_patches(svs_path, slide_level, valid_bounds, k, backbone, chief, 
                          anatomical_label=1, patch_size=224, device='cuda'):
    """
    Extract top K patches from a whole slide image based on CHIEF model predictions.
    
    Args:
        svs_path: Path to the slide file (.svs, .tiff, etc.)
        slide_level: Pyramid level to process (0 = highest resolution)
        valid_bounds: List of (x1, y1, x2, y2) tuples defining valid regions AT THE SPECIFIED SLIDE_LEVEL
        k: Number of top patches to return
        backbone: Pretrained feature extraction model
        chief: CHIEF model for patch probability prediction
        anatomical_label: Anatomical site label for CHIEF model
        patch_size: Size of each patch in pixels (default: 224)
        device: Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        tuple: (top_k_patches, heatmap)
            - top_k_patches: List of (x, y) tuples AT THE SPECIFIED SLIDE_LEVEL
            - heatmap: 2D numpy array of patch probabilities
    """
    # Initialize transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load WSI using OpenSlide
    slide = openslide.OpenSlide(svs_path)
    img_w, img_h = slide.level_dimensions[slide_level]
    downsample = slide.level_downsamples[slide_level]
    print(f"Processing slide at level {slide_level}: {img_w}x{img_h}, downsample: {downsample}")
    
    # Generate heatmap
    # Note: We scan in level-specific pixel space but convert to level 0 for read_region
    heatmap = []
    with torch.no_grad():
        for row_idx, top_level in enumerate(tqdm(range(0, img_h, patch_size), desc="Processing rows")):
            row_probs = []
            for col_idx, left_level in enumerate(range(0, img_w, patch_size)):
                # Convert level-specific coordinates to level 0 for OpenSlide
                left_level0 = int(left_level * downsample)
                top_level0 = int(top_level * downsample)
                
                # Read patch from slide (location in level 0, read at slide_level)
                patch = slide.read_region((left_level0, top_level0), slide_level, (patch_size, patch_size)).convert("RGB")
                
                if patch.size != (patch_size, patch_size):
                    row_probs.append(0.0)
                    continue
                
                patch_tensor = transform(patch).unsqueeze(0).to(device)
                feature = backbone(patch_tensor)
                result = chief.patch_probs(feature, torch.tensor([anatomical_label]).to(device))
                prob = result['patch_prob'].cpu().item()
                row_probs.append(prob)
            
            heatmap.append(row_probs)
    
    slide.close()
    heatmap = np.array(heatmap)
    
    # Get top K patches within valid bounds (valid_bounds are at slide_level)
    top_k_patches_indices = _get_top_k_patches_with_bounds(heatmap, k, valid_bounds, patch_size)
    
    # Convert to slide_level pixel coordinates
    top_k_patches = [(int(col * patch_size), int(row * patch_size)) 
                     for row, col in top_k_patches_indices]
    
    return top_k_patches, heatmap


def _get_top_k_patches_with_bounds(heatmap, k, valid_bounds, patch_size=224):
    """
    Returns the (row, col) indices of the top K highest value patches
    that fall within the specified valid bounds.
    
    Args:
        heatmap: 2D numpy array of patch probabilities
        k: Number of top patches to return
        valid_bounds: List of (x1, y1, x2, y2) tuples defining valid regions at slide_level
        patch_size: Size of each patch in pixels (default: 224)
    
    Returns:
        List of (row, col) tuples
    """
    # Create a mask for valid patches
    valid_mask = np.zeros_like(heatmap, dtype=bool)
    
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            # Convert (row, col) to slide_level pixel coordinates
            x = int(col * patch_size)
            y = int(row * patch_size)
            x2 = int(x + patch_size)
            y2 = int(y + patch_size)
            
            # Check if patch is inside any valid bound (valid_bounds are at slide_level)
            for bound_x1, bound_y1, bound_x2, bound_y2 in valid_bounds:
                if x >= bound_x1 and y >= bound_y1 and x2 <= bound_x2 and y2 <= bound_y2:
                    valid_mask[row, col] = True
                    break
    
    # Apply mask to heatmap (set invalid patches to -inf so they're not selected)
    masked_heatmap = heatmap.copy()
    masked_heatmap[~valid_mask] = -np.inf
    
    # Get top K from valid patches
    flat_indices = np.argsort(masked_heatmap.ravel())[::-1][:k]
    row_indices = flat_indices // heatmap.shape[1]
    col_indices = flat_indices % heatmap.shape[1]
    
    return list(zip(row_indices, col_indices))



def visualize_heatmap(heatmap, top_k_patches=None, patch_size=224, output_path="patch_probability_heatmap.png"):
    """
    Visualize the patch probability heatmap.
    
    Args:
        heatmap: 2D numpy array of patch probabilities
        top_k_patches: Optional list of (x, y) pixel coordinate tuples at slide_level
        patch_size: Size of each patch in pixels (default: 224)
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    
    if top_k_patches is not None:
        plt.title(f"Patch-Level Probability Heatmap with Top {len(top_k_patches)} Patches")
        for x_level, y_level in top_k_patches:
            # Convert slide_level coordinates to row/col indices for heatmap visualization
            col = int(x_level / patch_size)
            row = int(y_level / patch_size)
            rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1, 
                                 fill=False, edgecolor='green', linewidth=2)
            plt.gca().add_patch(rect)
    else:
        plt.title("Patch-Level Probability Heatmap")
    
    plt.colorbar(label="Probability")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap visualization to: {output_path}")


def save_annotated_slide(svs_path, slide_level, top_k_patches, patch_size=224, 
                         output_path=None):
    """
    Save a slide image with top K patches annotated as green rectangles.
    
    Args:
        svs_path: Path to the slide file
        slide_level: Pyramid level to save
        top_k_patches: List of (x, y) pixel coordinate tuples at slide_level
        patch_size: Size of each patch in pixels
        output_path: Path to save the annotated image. If None, auto-generates path.
    
    Returns:
        str: Path to the saved annotated image
    """
    # Load the slide and convert to image
    slide = openslide.OpenSlide(svs_path)
    width, height = slide.level_dimensions[slide_level]
    slide_image = slide.read_region((0, 0), slide_level, (width, height)).convert("RGB")
    slide.close()
    
    # Draw rectangles on the image
    draw = ImageDraw.Draw(slide_image)
    for x_level, y_level in top_k_patches:
        # Coordinates are already at slide_level
        x1 = int(x_level)
        y1 = int(y_level)
        x2 = int(x1 + patch_size)
        y2 = int(y1 + patch_size)
        draw.rectangle([x1, y1, x2, y2], outline='green', width=5)
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(svs_path))[0]
        output_path = f"{base_name}_level{slide_level}_topK_annotated.jpg"
    
    # Save the annotated image
    slide_image.save(output_path, "JPEG", quality=95)
    print(f"Saved annotated slide with Top {len(top_k_patches)} patches to: {output_path}")
    
    return output_path


def save_slide_as_jpg(svs_path, slide_level, output_path=None):
    """
    Load a slide at a given level and save it as JPG format.
    
    Args:
        svs_path: Path to the slide file
        slide_level: Pyramid level to read
        output_path: Path to save the JPG file. If None, auto-generates path.
    
    Returns:
        str: Path to the saved image
    """
    slide = openslide.OpenSlide(svs_path)
    width, height = slide.level_dimensions[slide_level]
    print(f"Loading slide at level {slide_level}: {width}x{height}")
    
    slide_image = slide.read_region((0, 0), slide_level, (width, height)).convert("RGB")
    slide.close()
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(svs_path))[0]
        output_path = f"{base_name}_level{slide_level}.jpg"
    
    slide_image.save(output_path, "JPEG", quality=95)
    print(f"Saved slide to: {output_path}")
    
    return output_path


# ==================== Main Execution ====================

if __name__ == "__main__":
    # Configuration
    print('yoyo')
    IMG_PATH = "example_svs/19.svs"
    PATCH_SIZE = 224
    ANATOMICAL_LABEL = 1
    SLIDE_LEVEL = 2
    TOP_K = 5
    VALID_BOUNDS = [(0, 600, 1600, 1000)]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {DEVICE}")
    
    # Load models
    print("Loading models....")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    backbone = ctranspath()
    backbone.head = nn.Identity()
    td = torch.load('./model_weight/CHIEF_CTransPath.pth', map_location=DEVICE)
    backbone.load_state_dict(td['model'], strict=True)
    backbone.eval().to(DEVICE)
    
    chief = CHIEF(size_arg="small", dropout=True, n_classes=2)
    chief.load_state_dict(torch.load('./model_weight/CHIEF_pretraining.pth', map_location=DEVICE), strict=True)
    chief.eval().to(DEVICE)
    
    # Get downsample factor
    slide_temp = openslide.OpenSlide(IMG_PATH)
    downsample = slide_temp.level_downsamples[SLIDE_LEVEL]
    slide_temp.close()
    
    # Extract top K patches
    tim = time.time()
    print(f"\nExtracting top {TOP_K} patches...")
    top_k_patches, heatmap = extract_top_k_patches(
        svs_path=IMG_PATH,
        slide_level=SLIDE_LEVEL,
        valid_bounds=VALID_BOUNDS,
        k=TOP_K,
        backbone=backbone,
        chief=chief,
        anatomical_label=ANATOMICAL_LABEL,
        patch_size=PATCH_SIZE,
        device=DEVICE
    )
    print('time taken:', time.time() - tim)
    
    print(f"Top {TOP_K} patches (slide level {SLIDE_LEVEL} coordinates): {top_k_patches}")
    
    # Visualize heatmap without annotations
    visualize_heatmap(heatmap, output_path="patch_probability_heatmap.png")
    
    # Visualize heatmap with top K patches marked
    visualize_heatmap(heatmap, top_k_patches, PATCH_SIZE, output_path="patch_probability_heatmap_top_k.png")
    
    # Save the slide as JPG for reference
    save_slide_as_jpg(IMG_PATH, SLIDE_LEVEL)
    
    # Save annotated slide with top K patches
    save_annotated_slide(IMG_PATH, SLIDE_LEVEL, top_k_patches, PATCH_SIZE)
    
    print("\nProcessing complete!")

"""
FastAPI endpoint that combines patch extraction with CHIEF API processing.
This API first extracts top N patches from SVS images, then processes them through CHIEF API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import os
import sys
from pathlib import Path
import logging
import requests

# Add histocartography to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Patch_Extraction_API', 'histocartography_test', 'histocartography'))

from PIL import Image, ImageDraw
import numpy as np
import openslide
import matplotlib.pyplot as plt
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Combined Patch Extraction & CHIEF Processing API",
    description="Extract patches from SVS images and process them through CHIEF API",
    version="1.0.0"
)

# Initialize cell graph generation components
logger.info("Initializing nuclei detector and feature extractor...")
nuclei_detector = NucleiExtractor()
feats_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72, resize_size=224)
knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)
logger.info("Initialization complete")

# CHIEF API configuration
CHIEF_API_URL = "http://localhost:8001/extract_patches"


class CombinedRequest(BaseModel):
    """Request model for combined patch extraction and CHIEF processing"""
    image_path: str = Field(..., description="Path to the SVS image file")
    svs_level: Optional[int] = Field(None, description="SVS level to use (0=highest resolution). If None, uses the smallest level.")
    top_n: int = Field(6, description="Number of top patches to extract based on nuclei density", ge=1)
    save_patches: bool = Field(False, description="Whether to save patches to disk")
    patch_size: int = Field(224, description="Patch size for CHIEF processing", ge=1)
    anatomical_label: int = Field(..., description="Anatomical label for CHIEF processing")
    top_k: int = Field(5, description="Top K patches for CHIEF processing", ge=1)


class CombinedResponse(BaseModel):
    """Response model forwarding CHIEF API response"""
    success: bool
    message: str
    patch_extraction_info: dict
    chief_response: Optional[dict] = None
    saved_patches_info: Optional[dict] = None


def svs_to_image(svs_path: str, level: int = None) -> Tuple[Image.Image, str, int]:
    """
    Load an SVS whole slide image and convert the specified level to PIL Image.
    
    Args:
        svs_path: Path to the input .svs file
        level: Level to use for conversion (0=highest resolution). If None, uses smallest level.
    
    Returns:
        Tuple of (PIL Image, level info string, actual level used)
    """
    logger.info(f"Loading SVS file: {svs_path}")
    
    slide = openslide.OpenSlide(svs_path)
    
    num_levels = slide.level_count
    level_info = f"Total levels: {num_levels}\n"
    for lvl in range(num_levels):
        dimensions = slide.level_dimensions[lvl]
        downsample = slide.level_downsamples[lvl]
        level_info += f"Level {lvl}: {dimensions[0]}x{dimensions[1]} (downsample: {downsample:.2f}x)\n"
    
    logger.info(level_info)
    
    if level is None:
        level = num_levels - 1
        logger.info(f"No level specified, using smallest level {level}")
    elif level < 0 or level >= num_levels:
        raise ValueError(f"Level {level} is out of range. Available levels: 0-{num_levels-1}")
    else:
        logger.info(f"Using specified level {level}")
    
    level_dims = slide.level_dimensions[level]
    logger.info(f"Level {level} dimensions: {level_dims[0]}x{level_dims[1]}")
    
    img = slide.read_region((0, 0), level, level_dims).convert('RGB')
    
    slide.close()
    return img, level_info, level


def extract_patches(image: Image.Image, top_n: int = 6) -> Tuple[List[dict], List[dict]]:
    """
    Extract top N patches from an image based on nuclei density.
    
    Args:
        image: PIL Image object
        top_n: Number of top patches to extract
    
    Returns:
        Tuple of (top patches list, non-selected patches list)
    """
    logger.info(f"Starting patch extraction (top_n={top_n})")
    
    image_array = np.array(image)
    
    logger.info("Detecting nuclei...")
    nuclei_map, nuclei_centers = nuclei_detector.process(image_array)
    logger.info(f"Detected {nuclei_centers.shape[0]} nuclei")
    
    if nuclei_centers.shape[0] <= 5:
        logger.warning(f"Insufficient nuclei detected ({nuclei_centers.shape[0]} ≤ 5). Skipping patch extraction.")
        return [], []
    
    logger.info("Extracting features...")
    features = feats_extractor.process(image_array, nuclei_map)
    
    logger.info("Building cell graph...")
    cell_graph = knn_graph_builder.process(nuclei_map, features)
    
    # Create uniform 3x3 grid (9 patches total)
    width, height = image.size
    width_range = np.linspace(0, width, 4, dtype=int)
    height_range = np.linspace(0, height, 4, dtype=int)
    
    # Extract patches from uniform grid (no overlap)
    logger.info("Extracting patches from uniform 3x3 grid...")
    patch_nuclei_centers = []
    patch_coordinates = []
    
    for i in range(len(width_range)-1):
        for j in range(len(height_range)-1):
            # Define uniform patch boundaries
            left = int(width_range[i])
            upper = int(height_range[j])
            right = int(width_range[i+1])
            lower = int(height_range[j+1])
            
            # Count nuclei in this patch
            center_list = []
            for center in nuclei_centers:
                if ((center[0] >= left) and (center[0] <= right) and 
                    (center[1] >= upper) and (center[1] <= lower)):
                    center_list.append(center)
            
            patch_nuclei_centers.append(center_list)
            patch_coordinates.append((left, upper, right, lower))
    
    # Calculate nuclei count per patch
    patch_center_length = [len(center) for center in patch_nuclei_centers]
    
    # Sort patches by nuclei count (descending)
    sorted_indices_desc = np.flip(np.argsort(patch_center_length))
    
    # Prepare top N patches and non-selected patches
    top_patch_info = []
    non_selected_patch_info = []
    actual_top_n = min(top_n, len(patch_coordinates))
    
    for patch_index in range(actual_top_n):
        idx = sorted_indices_desc[patch_index]
        coords = patch_coordinates[idx]
        nuclei_count = patch_center_length[idx]
        
        top_patch_info.append({
            'patch_number': patch_index + 1,
            'x1': coords[0],
            'y1': coords[1],
            'x2': coords[2],
            'y2': coords[3],
            'nuclei_count': nuclei_count
        })
        
        logger.info(f"Selected patch {patch_index+1}: coords={coords}, nuclei_count={nuclei_count}")
    
    # Collect non-selected patches
    for patch_index in range(actual_top_n, len(sorted_indices_desc)):
        idx = sorted_indices_desc[patch_index]
        coords = patch_coordinates[idx]
        nuclei_count = patch_center_length[idx]
        
        non_selected_patch_info.append({
            'x1': coords[0],
            'y1': coords[1],
            'x2': coords[2],
            'y2': coords[3],
            'nuclei_count': nuclei_count
        })
        
        logger.info(f"Non-selected patch: coords={coords}, nuclei_count={nuclei_count}")
    
    logger.info(f"Successfully extracted {actual_top_n} selected patches and {len(non_selected_patch_info)} non-selected patches")
    return top_patch_info, non_selected_patch_info


def save_chief_patches(svs_path: str, slide_level: int, chief_patches: List[dict], 
                       output_dir: str = "output_patches") -> List[str]:
    """
    Save patches from CHIEF API coordinates as individual JPG files.
    
    Args:
        svs_path: Path to the SVS file
        slide_level: SVS level to extract from
        chief_patches: List of dicts with x1, y1, x2, y2 coordinates at the specified slide_level
        output_dir: Directory to save patches
    
    Returns:
        List of saved file paths
    """
    logger.info(f"Saving {len(chief_patches)} patches to {output_dir}")
    
    # Create output directory and clean it
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove all existing images in the directory
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
            os.remove(file_path)
            logger.info(f"Removed existing file: {file}")
    
    # Open slide
    slide = openslide.OpenSlide(svs_path)
    downsample = slide.level_downsamples[slide_level]
    saved_files = []
    
    try:
        for idx, patch_info in enumerate(chief_patches, start=1):
            # Coordinates are at slide_level, need to convert to level 0 for read_region
            x1_level = patch_info['x1']
            y1_level = patch_info['y1']
            x2_level = patch_info['x2']
            y2_level = patch_info['y2']
            
            # Convert slide_level coordinates to level 0 for OpenSlide read_region
            x1_level0 = int(x1_level * downsample)
            y1_level0 = int(y1_level * downsample)
            
            # Calculate size at slide_level
            patch_width = x2_level - x1_level
            patch_height = y2_level - y1_level
            
            # Extract patch from slide (location in level 0, size in level pixels)
            # OpenSlide read_region: location in level 0, level to read from, size in level pixels
            patch = slide.read_region((x1_level0, y1_level0), slide_level, (patch_width, patch_height)).convert("RGB")
            
            # Save patch
            output_path = os.path.join(output_dir, f"patch_{idx}.jpg")
            patch.save(output_path, "JPEG", quality=95)
            saved_files.append(output_path)
            logger.info(f"Saved patch_{idx}.jpg at slide level {slide_level} coordinates ({x1_level}, {y1_level}, {x2_level}, {y2_level})")
    
    finally:
        slide.close()
    
    logger.info(f"Successfully saved {len(saved_files)} patches")
    return saved_files


def create_visualization(svs_path: str, slide_level: int, valid_bounds: List[List[int]], 
                        chief_patches: List[dict],
                        non_selected_patches: List[dict] = None,
                        output_dir: str = "output_patches") -> str:
    """
    Create a visualization with blue, green, and red rectangles for different patch types.
    All coordinates should be at the specified slide_level.
    
    Args:
        svs_path: Path to the SVS file
        slide_level: SVS level to visualize
        valid_bounds: List of [x1, y1, x2, y2] coordinates at slide_level (selected histocartography)
        chief_patches: List of dicts with x1, y1, x2, y2 coordinates at slide_level (CHIEF final)
        non_selected_patches: List of dicts with x1, y1, x2, y2 coordinates (non-selected histocartography)
        output_dir: Directory to save visualization
    
    Returns:
        Path to saved visualization
    """
    logger.info("Creating WSI visualization with patch annotations")
    
    # Load the slide at specified level
    slide = openslide.OpenSlide(svs_path)
    width, height = slide.level_dimensions[slide_level]
    slide_image = slide.read_region((0, 0), slide_level, (width, height)).convert("RGB")
    slide.close()
    
    # Draw on the image
    draw = ImageDraw.Draw(slide_image)
    
    # Draw red rectangles for NON-selected histocartography patches (first, so they're behind)
    if non_selected_patches:
        logger.info(f"Drawing {len(non_selected_patches)} red rectangles for non-selected histocartography patches")
        for patch_info in non_selected_patches:
            x1 = patch_info['x1']
            y1 = patch_info['y1']
            x2 = patch_info['x2']
            y2 = patch_info['y2']
            draw.rectangle([x1, y1, x2, y2], outline='red', width=5)
    
    # Draw blue rectangles for SELECTED histocartography patches - coordinates at slide_level
    logger.info(f"Drawing {len(valid_bounds)} blue rectangles for selected histocartography patches")
    for bound in valid_bounds:
        x1, y1, x2, y2 = bound
        draw.rectangle([x1, y1, x2, y2], outline='blue', width=5)
    
    # Draw green rectangles for CHIEF final patches - coordinates at slide_level
    logger.info(f"Drawing {len(chief_patches)} green rectangles for CHIEF final patches")
    for patch_info in chief_patches:
        x1 = patch_info['x1']
        y1 = patch_info['y1']
        x2 = patch_info['x2']
        y2 = patch_info['y2']
        draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "wsi_visualization.jpg")
    slide_image.save(output_path, "JPEG", quality=95)
    logger.info(f"Saved visualization to {output_path}")
    
    return output_path


def save_chief_heatmap(heatmap_data: List[List[float]], output_dir: str = "output_patches") -> str:
    """
    Save CHIEF heatmap as an image.
    
    Args:
        heatmap_data: 2D list of heatmap values
        output_dir: Directory to save heatmap
    
    Returns:
        Path to saved heatmap
    """
    logger.info("Saving CHIEF heatmap")
    
    # Convert to numpy array
    heatmap = np.array(heatmap_data)
    
    # Create heatmap visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title("CHIEF Patch Probability Heatmap")
    plt.colorbar(label="Probability")
    
    # Save heatmap
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "chief_heatmap.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved CHIEF heatmap to {output_path}")
    return output_path


def call_chief_api(image_path: str, patch_size: int, anatomical_label: int, 
                   slide_level: int, top_k: int, valid_bounds: List[List[int]]) -> dict:
    """
    Call the CHIEF API with extracted patch coordinates.
    
    Args:
        image_path: Path to the SVS file
        patch_size: Patch size for CHIEF processing
        anatomical_label: Anatomical label
        slide_level: SVS level (same as svs_level)
        top_k: Number of top patches for CHIEF
        valid_bounds: List of coordinate bounds from patch extraction
    
    Returns:
        CHIEF API response as dict
    """
    logger.info(f"Calling CHIEF API at {CHIEF_API_URL}")
    
    payload = {
        "image_path": image_path,
        "patch_size": patch_size,
        "anatomical_label": anatomical_label,
        "slide_level": slide_level,
        "top_k": top_k,
        "valid_bounds": valid_bounds
    }
    
    logger.info(f"CHIEF API payload: {payload}")
    
    try:
        response = requests.post(CHIEF_API_URL, json=payload, timeout=1800)  # 30 min timeout
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"CHIEF API returned successfully")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling CHIEF API: {str(e)}")
        raise HTTPException(
            status_code=502, 
            detail=f"Failed to call CHIEF API: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Combined Patch Extraction & CHIEF Processing API",
        "version": "1.0.0",
        "endpoints": {
            "/process": "POST - Extract patches and process through CHIEF API",
            "/health": "GET - Check API health status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}


@app.post("/process", response_model=CombinedResponse)
async def process_endpoint(request: CombinedRequest):
    """
    Extract top N patches from SVS image and process them through CHIEF API.
    
    This endpoint:
    1. Extracts top N patches based on nuclei density
    2. Uses those patch coordinates as valid_bounds for CHIEF API
    3. Returns the CHIEF API response
    
    Note: Processing may take up to 30 minutes for large images.
    
    Args:
        request: CombinedRequest with all parameters
    
    Returns:
        CombinedResponse with patch extraction info and CHIEF API response
    """
    try:
        logger.info(f"Received combined processing request: {request}")
        
        # Validate file exists
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail=f"Image file not found: {request.image_path}")
        
        # Validate file type
        file_ext = Path(request.image_path).suffix.lower()
        if file_ext != '.svs':
            raise HTTPException(
                status_code=400, 
                detail=f"Only SVS files are supported. Got: {file_ext}"
            )
        
        # Step 1: Extract patches
        logger.info("Step 1: Extracting patches...")
        image, level_info, actual_slide_level = svs_to_image(request.image_path, request.svs_level)
        patch_info_list, non_selected_patches = extract_patches(image=image, top_n=request.top_n)
        
        if not patch_info_list:
            return CombinedResponse(
                success=False,
                message="Insufficient nuclei detected for patch extraction",
                patch_extraction_info={
                    "total_patches_extracted": 0,
                    "coordinates": []
                },
                chief_response=None
            )
        
        # Convert patch coordinates to valid_bounds format for CHIEF API
        # valid_bounds expects [[x1, y1, x2, y2], ...] at the slide_level
        # Coordinates are already in level-specific space
        valid_bounds = [[
            p['x1'],
            p['y1'],
            p['x2'],
            p['y2']
        ] for p in patch_info_list]
        
        logger.info(f"Extracted {len(patch_info_list)} patches at slide level {actual_slide_level}")
        
        # Step 2: Call CHIEF API with extracted coordinates
        logger.info("Step 2: Calling CHIEF API...")
        chief_response = call_chief_api(
            image_path=request.image_path,
            patch_size=request.patch_size,
            anatomical_label=request.anatomical_label,
            slide_level=actual_slide_level,
            top_k=request.top_k,
            valid_bounds=valid_bounds
        )
        
        # Step 3: Save patches if requested
        saved_patches_info = None
        if request.save_patches:
            logger.info("Step 3: Saving patches and creating visualization...")
            
            # Extract CHIEF patch coordinates from response
            chief_patches = chief_response.get('top_k_patches', [])
            chief_heatmap = chief_response.get('heatmap', [])
            
            if chief_patches:
                # All coordinates are now at slide_level - no conversion needed
                
                # Save patches using slide_level coordinates
                saved_files = save_chief_patches(
                    svs_path=request.image_path,
                    slide_level=actual_slide_level,
                    chief_patches=chief_patches,  # slide_level coordinates
                    output_dir="output_patches"
                )
                
                # Save CHIEF heatmap
                heatmap_path = save_chief_heatmap(
                    heatmap_data=chief_heatmap,
                    output_dir="output_patches"
                )
                
                # Create visualization using slide_level coordinates
                valid_bounds_level = [[p['x1'], p['y1'], p['x2'], p['y2']] for p in patch_info_list]
                viz_path = create_visualization(
                    svs_path=request.image_path,
                    slide_level=actual_slide_level,
                    valid_bounds=valid_bounds_level,  # slide_level coordinates (blue - selected)
                    chief_patches=chief_patches,  # slide_level coordinates (green - final)
                    non_selected_patches=non_selected_patches,  # slide_level coordinates (red - not selected)
                    output_dir="output_patches"
                )
                
                saved_patches_info = {
                    "patches_saved": len(saved_files),
                    "patch_files": saved_files,
                    "visualization_file": viz_path,
                    "heatmap_file": heatmap_path
                }
                logger.info(f"Saved {len(saved_files)} patches, visualization, and heatmap")
            else:
                logger.warning("No patches returned from CHIEF API to save")
        
        return CombinedResponse(
            success=True,
            message="Successfully processed patches through CHIEF API",
            patch_extraction_info={
                "total_patches_extracted": len(patch_info_list),
                "coordinates": patch_info_list
            },
            chief_response=chief_response,
            saved_patches_info=saved_patches_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run on port 8003 to avoid conflicts with other APIs
    # Port 8000: (possibly other services)
    # Port 8001: CHIEF API
    # Port 8002: Patch Extraction API
    # Port 8003: Combined API
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8003,
        timeout_keep_alive=1800,  # 30 minutes
        log_level="info"
    )

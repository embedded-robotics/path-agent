"""
FastAPI endpoint for patch extraction from SVS or flat pathology images.
This API provides endpoints to extract top N patches from SVS, JPG/JPEG, TIFF, or PNG files.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import os
import sys
from pathlib import Path
import logging

# Add histocartography to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'histocartography_test', 'histocartography'))

from PIL import Image
import numpy as np
import openslide
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with custom timeout hint
app = FastAPI(
    title="Pathology Patch Extraction API",
    description="Extract top N patches from SVS or flat pathology images based on nuclei density",
    version="1.0.0"
)

# Initialize cell graph generation components
logger.info("Initializing nuclei detector and feature extractor...")
nuclei_detector = NucleiExtractor()
feats_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72, resize_size=224)
knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)
logger.info("Initialization complete")


class PatchRequest(BaseModel):
    """Request model for patch extraction"""
    image_path: str = Field(..., description="Path to the input image file (.svs, .jpg/.jpeg, .tif/.tiff, .png)")
    svs_level: Optional[int] = Field(None, description="SVS level to use (0=highest resolution). If None, uses the smallest level. Ignored for non-SVS files, which are treated as already at the required level.")
    top_n: int = Field(6, description="Number of top patches to extract based on nuclei density", ge=1)
    save_patches: bool = Field(False, description="Whether to save patches to disk")
    output_dir: Optional[str] = Field("output_patches", description="Directory to save patches (if save_patches=True)")


class PatchCoordinate(BaseModel):
    """Model for patch coordinates"""
    patch_number: int
    x1: int
    y1: int
    x2: int
    y2: int
    nuclei_count: int
    
    
class PatchResponse(BaseModel):
    """Response model for patch extraction"""
    success: bool
    message: str
    image_path: str
    total_patches_extracted: int
    coordinates: List[PatchCoordinate]
    output_dir: Optional[str] = None


def svs_to_image(svs_path: str, level: int = None) -> Tuple[Image.Image, str]:
    """
    Load an SVS whole slide image and convert the specified level to PIL Image.
    
    Args:
        svs_path: Path to the input .svs file
        level: Level to use for conversion (0=highest resolution). If None, uses smallest level.
    
    Returns:
        Tuple of (PIL Image, level info string)
    """
    logger.info(f"Loading SVS file: {svs_path}")
    
    # Load the SVS file
    slide = openslide.OpenSlide(svs_path)
    
    # Get level information
    num_levels = slide.level_count
    level_info = f"Total levels: {num_levels}\n"
    for lvl in range(num_levels):
        dimensions = slide.level_dimensions[lvl]
        downsample = slide.level_downsamples[lvl]
        level_info += f"Level {lvl}: {dimensions[0]}x{dimensions[1]} (downsample: {downsample:.2f}x)\n"
    
    logger.info(level_info)
    
    # Use provided level or default to smallest level
    if level is None:
        level = num_levels - 1
        logger.info(f"No level specified, using smallest level {level}")
    elif level < 0 or level >= num_levels:
        raise ValueError(f"Level {level} is out of range. Available levels: 0-{num_levels-1}")
    else:
        logger.info(f"Using specified level {level}")
    
    level_dims = slide.level_dimensions[level]
    logger.info(f"Level {level} dimensions: {level_dims[0]}x{level_dims[1]}")
    
    # Read the image at the specified level (RGBA -> RGB)
    img = slide.read_region((0, 0), level, level_dims).convert('RGB')
    
    slide.close()
    return img, level_info


def extract_patches(image: Image.Image, top_n: int = 6, save_patches: bool = False, 
                    output_dir: str = "output_patches") -> Tuple[List[dict], List[Image.Image]]:
    """
    Extract top N patches from an image based on nuclei density.
    
    Args:
        image: PIL Image object
        top_n: Number of top patches to extract
        save_patches: Whether to save patches to disk
        output_dir: Directory to save patches
    
    Returns:
        Tuple of (list of patch info dicts, list of patch images)
    """
    logger.info(f"Starting patch extraction (top_n={top_n})")
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Detect nuclei
    logger.info("Detecting nuclei...")
    nuclei_map, nuclei_centers = nuclei_detector.process(image_array)
    logger.info(f"Detected {nuclei_centers.shape[0]} nuclei")
    
    # Only process if more than 5 nuclei are detected
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
    image_patches = []
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
            
            image_patches.append(image.crop((left, upper, right, lower)))
            patch_nuclei_centers.append(center_list)
            patch_coordinates.append((left, upper, right, lower))
    
    # Calculate nuclei count per patch
    patch_center_length = [len(center) for center in patch_nuclei_centers]
    
    # Sort patches by nuclei count (descending)
    sorted_indices_desc = np.flip(np.argsort(patch_center_length))
    
    # Prepare top N patches
    top_patch_info = []
    top_patch_images = []
    actual_top_n = min(top_n, len(image_patches))
    
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
        top_patch_images.append(image_patches[idx])
        
        logger.info(f"Patch {patch_index+1}: coords={coords}, nuclei_count={nuclei_count}")
    
    # Save patches if requested
    if save_patches:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Saving patches to {output_dir}...")
        for i, (patch_img, patch_info) in enumerate(zip(top_patch_images, top_patch_info), 1):
            save_path = os.path.join(output_dir, f"patch_{i}.png")
            patch_img.save(save_path)
            logger.info(f"Saved: {save_path}")
    
    logger.info(f"Successfully extracted {actual_top_n} patches")
    return top_patch_info, top_patch_images


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Pathology Patch Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "/extract_patches": "POST - Extract patches from SVS or JPG images",
            "/health": "GET - Check API health status"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}


@app.post("/extract_patches", response_model=PatchResponse)
async def extract_patches_endpoint(request: PatchRequest):
    """
    Extract top N patches from an SVS or flat pathology image.
    
    Note: Processing may take up to 15 minutes for large images.
    The endpoint will not timeout during processing.
    
    Args:
        request: PatchRequest containing image_path, svs_level (SVS only), top_n, save_patches, output_dir
    
    Returns:
        PatchResponse with coordinates and metadata
    """
    try:
        logger.info(f"Received request: {request}")
        
        # Validate file exists
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail=f"Image file not found: {request.image_path}")
        
        # Determine file type
        file_ext = Path(request.image_path).suffix.lower()
        
        if file_ext == '.svs':
            logger.info("Processing SVS file...")
            image, level_info = svs_to_image(request.image_path, request.svs_level)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            logger.info("Processing flat image file (already at requested level)...")
            image = Image.open(request.image_path).convert('RGB')
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_ext}. Supported formats: .svs, .jpg, .jpeg, .tif, .tiff, .png"
            )
        
        # Extract patches
        patch_info_list, patch_images = extract_patches(
            image=image,
            top_n=request.top_n,
            save_patches=request.save_patches,
            output_dir=request.output_dir
        )
        
        if not patch_info_list:
            return PatchResponse(
                success=False,
                message="Insufficient nuclei detected for patch extraction",
                image_path=request.image_path,
                total_patches_extracted=0,
                coordinates=[],
                output_dir=request.output_dir if request.save_patches else None
            )
        
        # Convert to response model
        coordinates = [PatchCoordinate(**info) for info in patch_info_list]
        
        return PatchResponse(
            success=True,
            message="Patches extracted successfully",
            image_path=request.image_path,
            total_patches_extracted=len(coordinates),
            coordinates=coordinates,
            output_dir=request.output_dir if request.save_patches else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn with custom timeout settings
    # timeout_keep_alive=1800 allows 30 minutes for long-running operations
    # Port 8002 to avoid conflict with other API on port 8000
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8002,
        timeout_keep_alive=1800,  # 30 minutes
        log_level="info"
    )

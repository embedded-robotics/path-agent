from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Tuple
import torch
import torch.nn as nn
from models.ctran import ctranspath
from models.CHIEF import CHIEF
import os
import sys
import platform

# Import the extraction function from the existing script
from chief_heatmap import extract_top_k_patches

app = FastAPI(title="CHIEF Patch Extraction API")

# Global model variables
backbone = None
chief = None
DEVICE = None

def get_smart_path(full_windows_path):
    if os.path.exists(full_windows_path):
        return full_windows_path

    in_container = platform.system() == "Linux" or os.path.exists('/.dockerenv')
    if not in_container:
        return full_windows_path

    normalized = full_windows_path.replace('\\', '/')

    for marker in ["svs_examples", "wsi_examples"]:
        if marker in normalized:
            relative_part = normalized.split(marker, 1)[-1].lstrip('/\\')
            container_path = os.path.join("/data", relative_part).replace('\\', '/')
            if os.path.exists(container_path):
                return container_path

    fallback_path = os.path.join("/data", os.path.basename(normalized)).replace('\\', '/')
    if os.path.exists(fallback_path):
        return fallback_path

    return full_windows_path


class PatchExtractionRequest(BaseModel):
    image_path: str = Field(..., description="Path to the slide file (.svs, .tiff, etc.)")
    patch_size: int = Field(224, description="Size of each patch in pixels")
    anatomical_label: int = Field(1, description="Anatomical site label for CHIEF model")
    slide_level: int = Field(2, description="Pyramid level to process (0 = highest resolution)")
    top_k: int = Field(5, description="Number of top patches to return")
    valid_bounds: List[Tuple[int, int, int, int]] = Field(
        [(0, 600, 1600, 1000)],
        description="List of (x1, y1, x2, y2) tuples defining valid regions at the specified slide_level"
    )


class PatchCoordinate(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class PatchExtractionResponse(BaseModel):
    top_k_patches: List[PatchCoordinate] = Field(
        ..., 
        description="List of patch coordinates with x1, y1, x2, y2 at the specified slide_level"
    )
    heatmap: List[List[float]] = Field(
        ...,
        description="2D heatmap array of patch probabilities"
    )
    slide_level: int = Field(..., description="The slide level used for coordinates")
    message: str = Field("Success", description="Status message")


@app.on_event("startup")
async def load_models():
    """Load CHIEF models on startup"""
    global backbone, chief, DEVICE
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading models on device: {DEVICE}")
    
    # Load backbone
    backbone = ctranspath()
    backbone.head = nn.Identity()
    td = torch.load('./model_weight/CHIEF_CTransPath.pth', map_location=DEVICE)
    backbone.load_state_dict(td['model'], strict=True)
    backbone.eval().to(DEVICE)
    
    # Load CHIEF model
    chief = CHIEF(size_arg="small", dropout=True, n_classes=2)
    chief.load_state_dict(
        torch.load('./model_weight/CHIEF_pretraining.pth', map_location=DEVICE), 
        strict=True
    )
    chief.eval().to(DEVICE)
    
    print("Models loaded successfully!")


@app.post("/extract_patches", response_model=PatchExtractionResponse)
async def extract_patches(request: PatchExtractionRequest):
    """
    Extract top K patches from a whole slide image based on CHIEF model predictions.
    
    Args:
        request: PatchExtractionRequest containing all parameters
    
    Returns:
        PatchExtractionResponse with top K patch coordinates
    """
    global backbone, chief, DEVICE
    
    # Validate that models are loaded
    if backbone is None or chief is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Validate that the image file exists
    
    
    req_image_path = request.image_path
    
    req_image_path = get_smart_path(req_image_path)
    
    print(f"Checking if file exists: {req_image_path}")
    print(f"os.path.exists result: {os.path.exists(req_image_path)}")
    print(f"Current working directory: {os.getcwd()}")
    if not os.path.exists(req_image_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Image file not found: {req_image_path}"
        )
    
    try:
        print(f"Starting extraction for {req_image_path}...")
        sys.stdout.flush()
        
        # Extract top K patches
        top_k_patches, heatmap = extract_top_k_patches(
            svs_path=req_image_path,
            slide_level=request.slide_level,
            valid_bounds=request.valid_bounds,
            k=request.top_k,
            backbone=backbone,
            chief=chief,
            anatomical_label=request.anatomical_label,
            patch_size=request.patch_size,
            device=DEVICE
        )
        
        print(f"Extraction complete! Found {len(top_k_patches)} patches")
        sys.stdout.flush()
        
        # Convert patches to x1, y1, x2, y2 format (coordinates are at slide_level)
        patch_coords = [
            PatchCoordinate(
                x1=x, 
                y1=y, 
                x2=x + request.patch_size, 
                y2=y + request.patch_size
            )
            for x, y in top_k_patches
        ]
        
        # Convert heatmap to list format for JSON serialization
        heatmap_list = heatmap.tolist()
        
        return PatchExtractionResponse(
            top_k_patches=patch_coords,
            heatmap=heatmap_list,
            slide_level=request.slide_level,
            message=f"Successfully extracted {len(top_k_patches)} patches at slide level {request.slide_level}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during patch extraction: {str(e)}"
        )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CHIEF Patch Extraction API is running",
        "device": DEVICE,
        "models_loaded": backbone is not None and chief is not None
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": backbone is not None and chief is not None
    }


if __name__ == "__main__":
    import uvicorn
    # Use log_config=None to allow tqdm and print to work properly
    uvicorn.run(app, host="0.0.0.0", port=8001, log_config=None)

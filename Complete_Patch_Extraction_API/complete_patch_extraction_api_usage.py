"""
Combined API Usage Example - Extract patches and process through CHIEF API
Update the parameters to match your requirements
"""

import requests
import json
import time
import os
from pathlib import Path

# Combined API endpoint
API_URL = "http://localhost:8003"
DEFAULT_SVS_IMAGE_PATH = str((Path(__file__).resolve().parents[1] / "svs_examples" / "19.svs").resolve())

# Request parameters
payload = {
    "image_path": os.getenv("SVS_IMAGE_PATH", DEFAULT_SVS_IMAGE_PATH),
    "svs_level": 2,                               # SVS level (same as slide_level for CHIEF)
    "top_n": 3,                                   # Number of top patches to extract
    "save_patches": True,                         # Whether to save patches to disk
    "patch_size": 224,                            # Patch size for CHIEF processing
    "anatomical_label": 1,                        # Anatomical label for CHIEF
    "top_k": 5                                    # Top K patches for CHIEF processing
}

# Make the request
print("=" * 60)
print("Combined API - Patch Extraction + CHIEF Processing")
print("=" * 60)
print(f"\nCalling API at {API_URL}/process...")
print(f"Image: {payload['image_path']}")
print(f"SVS Level: {payload['svs_level']}")
print(f"Top N patches to extract: {payload['top_n']}")
print(f"CHIEF Top K: {payload['top_k']}")
print()

start_time = time.time()
response = requests.post(f"{API_URL}/process", json=payload)
elapsed_time = time.time() - start_time

print(f"Request completed in {elapsed_time:.2f} seconds")
print("=" * 60)

# Check response
if response.status_code == 200:
    result = response.json()

    print('result')
    print(result)

    
    print(f"\n✓ Success: {result['success']}")
    print(f"Message: {result['message']}")
    
    # Print patch extraction info
    print("\n" + "=" * 60)
    print("PATCH EXTRACTION RESULTS")
    print("=" * 60)
    patch_info = result['patch_extraction_info']
    print(f"Total patches extracted: {patch_info['total_patches_extracted']}")
    print("\nExtracted patch coordinates:")
    for coord in patch_info['coordinates']:
        print(f"  Patch {coord['patch_number']}: "
              f"({coord['x1']}, {coord['y1']}) to ({coord['x2']}, {coord['y2']}) "
              f"- {coord['nuclei_count']} nuclei")
    
    # Print CHIEF API response
    print("\n" + "=" * 60)
    print("CHIEF API RESULTS")
    print("=" * 60)
    if result['chief_response']:
        chief_resp = result['chief_response']
        print(f"CHIEF API processed successfully")
        
        # Print CHIEF response structure
        if 'top_k_patches' in chief_resp:
            patches = chief_resp['top_k_patches']
            print(f"CHIEF returned {len(patches)} patches:")
            for idx, patch in enumerate(patches, start=1):
                print(f"  Patch {idx}: "
                      f"x1={patch.get('x1')}, y1={patch.get('y1')}, "
                      f"x2={patch.get('x2')}, y2={patch.get('y2')}")
        else:
            print("Full CHIEF response:")
            print(json.dumps(chief_resp, indent=2))
    else:
        print("No CHIEF response returned")
    
    # Print saved patches info if available
    if result.get('saved_patches_info'):
        print("\n" + "=" * 60)
        print("SAVED PATCHES INFO")
        print("=" * 60)
        saved_info = result['saved_patches_info']
        print(f"Patches saved: {saved_info.get('patches_saved', 0)}")
        print(f"Visualization file: {saved_info.get('visualization_file', 'N/A')}")
        print(f"Heatmap file: {saved_info.get('heatmap_file', 'N/A')}")
        if 'patch_files' in saved_info:
            print(f"\nSaved patch files:")
            for pf in saved_info['patch_files']:
                print(f"  - {pf}")
    
    print("\n" + "=" * 60)
    
else:
    print(f"\n✗ Error {response.status_code}")
    print(f"Response: {response.text}")
    try:
        error_detail = response.json()
        print(f"Details: {json.dumps(error_detail, indent=2)}")
    except:
        pass

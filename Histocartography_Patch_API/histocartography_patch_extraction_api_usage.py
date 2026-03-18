"""
Simple API usage example.
Supported input types: .svs, .jpg/.jpeg, .tif/.tiff, .png
For .svs, set svs_level as needed.
For flat images (.jpg/.tif/.png), svs_level is ignored.
"""

import requests
import json

API_URL = "http://localhost:8002"

# Call the API
print('calling API...')
response = requests.post(
    f"{API_URL}/extract_patches",
    json={
        "image_path": "E:/CHIEF_Test/CHIEF/19.svs",
        "svs_level": 2,
        "top_n": 3,
        "save_patches": False
    }
)

# Print results
result = response.json()
print(f"Success: {result['success']}")
print(f"Patches: {result['total_patches_extracted']}")
print("\nCoordinates:")
coordinates = result['coordinates']
for coord in coordinates:
    print(f"  Patch {coord['patch_number']}: ({coord['x1']}, {coord['y1']}) to ({coord['x2']}, {coord['y2']}) - {coord['nuclei_count']} nuclei")

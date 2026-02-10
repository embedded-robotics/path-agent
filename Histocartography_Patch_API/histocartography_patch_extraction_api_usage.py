"""
Simple API usage example - Extract patches from SVS file
Update the image path to match your file
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

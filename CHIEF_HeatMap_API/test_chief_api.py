import requests
import time

# API endpoint
url = "http://localhost:8001/extract_patches"

# Request payload
payload = {
    "image_path": "E:/CHIEF_Test/CHIEF/19.svs", 
    "patch_size": 224,
    "anatomical_label": 1,
    "slide_level": 2,
    "top_k": 5,
    "valid_bounds": [[0, 600, 1600, 1000]]
}

# Make the request
print('making request')
tim = time.time()
response = requests.post(url, json=payload)
print('time taken:', time.time() - tim)
print('request made')
print(response)

# Print results
if response.status_code == 200:
    result = response.json()
    print(result)
    print(f"Success! Found {len(result['coordinates'])} patches:")
    for patch in result['coordinates']:
        print(f"  Patch {patch['patch_number']}: x1={patch['x1']}, y1={patch['y1']}, x2={patch['x2']}, y2={patch['y2']}, nuclei={patch['nuclei_count']}")
else:
    print(f"Error {response.status_code}: {response.text}")



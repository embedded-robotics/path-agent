import requests

# API endpoint
url = "http://localhost:8001/extract_patches"

# Request payload
payload = {
    "image_path": "E:/Quilt1M/Gitti/path-agent/wsi_examples/19_level2.jpg",
    "patch_size": 224,
    "anatomical_label": 1,
    "slide_level": 2,
    "top_k": 5,
    "valid_bounds": [[0, 600, 1600, 1000]]
}

# Make the request
print('making request')
response = requests.post(url, json=payload)

# Print results
if response.status_code == 200:
    result = response.json()
    print(f"Success! Found {len(result['top_k_patches'])} patches:")
    for i, patch in enumerate(result['top_k_patches'], 1):
        print(
            f"  Patch {i}: x1={patch['x1']}, y1={patch['y1']}, "
            f"x2={patch['x2']}, y2={patch['y2']}"
        )
else:
    print(f"Error {response.status_code}: {response.text}")

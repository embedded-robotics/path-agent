import json
import requests

API_URL = "http://127.0.0.1:8000/match"

def main():
    payload = {
        "image_path": r"brst_site_image.JPG",   # <-- change this
        "queries": [
            "lung histopathology image",
            "breast histopathology image",
            "kidney histopathology image"
        ],
        "anatomical_site": "breast",  # <-- change to a valid key in your model_names
        "k": 2
    }

    resp = requests.post(API_URL, json=payload, timeout=300)
    print("Status:", resp.status_code)

    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    main()

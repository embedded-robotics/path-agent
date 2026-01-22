import json
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_match_query_list():
    print("\n--- Testing /match_query_list ---")
    url = f"{BASE_URL}/match_query_list"
    
    payload = {
        "image_path": r"images/brst_lbls.jpg",  # <-- Update with valid path
        "queries": [
            "breast tissue",
            "liver tissue",
            "colon tissue"
        ],
        "anatomical_site": "breast",
        "k": 2
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code == 200:
            print("Success!")
            print(json.dumps(resp.json(), indent=2))
        else:
            print("Failed:", resp.status_code)
            print(resp.text)
    except Exception as e:
        print(f"Error: {e}")


def test_retrieve_captions():
    print("\n--- Testing /retrieve_captions ---")
    url = f"{BASE_URL}/retrieve_captions"
    
    payload = {
        "images": [
            r"images/brst_site_image.jpg", # <-- Update with valid path
            r"images/brst_lbls.jpg"
        ],
        "anatomical_site": "breast",
        "k": 3
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code == 200:
            print("Success!")
            print(json.dumps(resp.json(), indent=2))
        else:
            print("Failed:", resp.status_code)
            print(resp.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_match_query_list()
    test_retrieve_captions()
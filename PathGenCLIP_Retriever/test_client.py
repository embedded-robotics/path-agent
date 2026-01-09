import json
import requests

API_URL = "http://127.0.0.1:8000/match"

def main():
    payload = {
        "image_path": r"brst_lbls.jpg",   # <-- change this
        "queries": [
            "Comparison with a normal breast section shows presence of ductal and lobular structures in the fibro-adipose stroma. if we zoom in a little bit we can see some irregular nests of cells within the fibrous stroma. So actually the original ductal and lobular architecture of the breast is lost here. Just for comparison, here is a section taken from a normal breast and again you can see the fibro-adipose stroma and here are the ductal and the lobular structures. So this architecture is not seen in the core biopsy.",
            "Clusters of cells called lobules are present, which contain ducts and glandular tissue or acini in an active mammary gland. by a vast network of dense, irregular connective tissue. And within it, we can see patches of adipocytes, or adipose tissue. And also, we can see these clusters of cells. And these clusters of cells are important because these are what are called lobules. And within each one of these lobules, we're actually going to find a series of ducts. And in an active mammary gland, we'd see a set of ducts as well as glandular tissue or acini.",
            "Presence of fissured and cracked amorphous pink material in the dermis associated with red cell extravasation, suggestive of amyloid deposition. Thioflavin-T or congo red stains are better for immunoglobulin derived amyloid. Colloid milium stains weakly for amyloid and has a background of solar elastosis. okay Yeah, that is fissured and cracked. So fissured amorphous pink material and cracked sitting in the dermis associated with red cell extravasation. So amyloid would be absolutely your thought. This was a patient with Waldenstroms and deposition, but amyloid, thioflavin T would be a great stain"
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

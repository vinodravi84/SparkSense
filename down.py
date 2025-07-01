import json
import random
import requests
import time

# 1) pip install requests
# 2) Sign up at https://pixabay.com/accounts/register/ and get your free API key

API_KEY     = "YOUR_PIXABAY_API_KEY"
PIXABAY_URL = "https://pixabay.com/api/"

def fetch_pixabay_image(query):
    params = {
        "key": API_KEY,
        "q": query,
        "image_type": "photo",
        "per_page": 10,
        "safesearch": "true"
    }
    resp = requests.get(PIXABAY_URL, params=params)
    data = resp.json()
    hits = data.get("hits", [])
    if not hits:
        return None
    return random.choice(hits)["webformatURL"]

def update_images(input_json, output_json):
    with open(input_json, "r") as f:
        data = json.load(f)

    # iterate both Men and Women lists
    for section in ("Men", "Women"):
        for p in data[section]:
            name = p["name"]
            query = f"{name} clothing"
            url = fetch_pixabay_image(query)
            if url:
                p["image"] = url
            else:
                print(f"⚠️ No Pixabay result for '{query}'")
            time.sleep(0.2)  # gentle on the API

    # write updated JSON
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    update_images(
        "products_150_split.json",
        "products_150_pixabay.json"
    )
    print("✅ Done! See products_150_pixabay.json")

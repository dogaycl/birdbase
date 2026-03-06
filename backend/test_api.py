import requests

url = "http://localhost:8000/predict/"
file_path = "/Users/dogayucel/.gemini/antigravity/brain/69f1f323-e141-4865-9eb1-badc4100382a/sample_eagle_1771577662056.png"

try:
    with open(file_path, "rb") as image_file:
        files = {"file": ("sample_eagle.png", image_file, "image/png")}
        response = requests.post(url, files=files)
        print("Status Code:", response.status_code)
        print("Response JSON:")
        print(response.json())
except Exception as e:
    print("Error:", e)

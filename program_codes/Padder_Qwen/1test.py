# # Checking Paddle ocr accessible or not

# import requests

# url = "http://10.68.0.39:9006/v1/models"

# try:
#     response = requests.get(url, timeout=10)

#     print("Status Code:", response.status_code)
#     print("Response:", response.text)

#     if response.status_code == 200:
#         print("PaddleOCR endpoint is accessible!")
#     else:
#         print("Endpoint responded, but returned an error.")

# except requests.exceptions.RequestException as e:
#     print("PaddleOCR endpoint is not accessible.")
#     print("Error:", e)


# Checking the qwen model
import requests

url = "http://10.68.0.39:9005/v1/models"

try:
    response = requests.get(url, timeout=10)

    print("Status Code:", response.status_code)
    print("Response:", response.text)

    if response.status_code == 200:
        for model in response.json().get("data", []):
            print("Available model ID:", model["id"])

except requests.exceptions.RequestException as e:
    print("Qwen endpoint error:", e)
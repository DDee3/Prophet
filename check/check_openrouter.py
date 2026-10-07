import requests

response = requests.get("https://openrouter.ai/api/v1/models")
for m in response.json().get("data", []):
    if ":free" in m["id"].lower():
        print(m["id"])
import requests
import os
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = os.getenv("HF_TOKEN")
api_url = "https://api-inference.huggingface.co/"
api_url += f"pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}
def query(texts):
  response = requests.post(api_url, headers=headers,
  json={"inputs": texts, 
  "options":{"wait_for_model":True}})
  return response.json()
texts = ["myszka miki",
            "ser",
            "pułapka",
            "szczur",
            "ratatuj"
            "autobus",
            "samolot",
            "statek"]
output = query(texts)
print(output)


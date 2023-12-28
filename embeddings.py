import requests

model_id = "sentence-transformers/all-MiniLM-L6-v2"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer "}


def text_embedding(texts):
    response = requests.post(
        api_url,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}},
    )
    return response.json()

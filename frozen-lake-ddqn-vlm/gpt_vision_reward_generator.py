import base64
import json

import requests
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

open_ai_api_key = os.environ["OPEN_AI_API_KEY"]

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {open_ai_api_key}",
}


def get_state_embedding(question_text, image_b64):
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    gpt_response = response.json()["choices"][0]["message"]["content"]
    print(gpt_response)

    data = {"input": gpt_response, "model": "text-embedding-3-large"}

    response = requests.post(
        "https://api.openai.com/v1/embeddings", headers=headers, data=json.dumps(data)
    )

    embedding = np.array(response.json()["data"][0]["embedding"])
    return embedding


def get_embedding_distance(current_state, goal_state):
    return np.dot(current_state, goal_state)


if __name__ == "__main__":
    with open("./test/image.png", "rb") as image_file:
        encoded_file = base64.b64encode(image_file.read()).decode("utf-8")

    original_state = get_state_embedding(
        "which row and column is the player in?",
        encoded_file,
    )

    with open("./test/image2.png", "rb") as image_file:
        encoded_file = base64.b64encode(image_file.read()).decode("utf-8")

    new_state = get_state_embedding(
        "which row and column is the player in?",
        encoded_file,
    )

    print(get_embedding_distance(original_state, new_state))

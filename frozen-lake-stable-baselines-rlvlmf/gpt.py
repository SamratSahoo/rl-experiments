import base64
import json
import re

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


def generate_analysis(image1, image2, objective):
    image1 = base64.b64encode(image1).decode("utf-8")
    image2 = base64.b64encode(image2).decode("utf-8")

    IMAGE1_PROMPT = f"""Consider the following two images: 
                    Image 1:"""
    IMAGE2_PROMPT = f"""Image 2:"""

    ANALYSIS_PROMPT = f"""        
        1. What is shown in Image 1? 
        2. What is shown in Image 2? 
        3. The goal is to {objective}. Is there any difference between Image 1 and Image 2 in terms of how close the agent has gotten towards achieving the goal?
    """

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE1_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image1}"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE2_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image2}"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            },
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    analysis = response.json()["choices"][0]["message"]["content"]
    return analysis


def generate_label(vlm_analysis, objective):
    LABEL_TEMPLATE = f"""
        Based on the text responses below to the questions:
            1. What is shown in Image 1? 
            2. What is shown in Image 2? 
            3. The goal is to {objective}. Is there any difference between Image 1 and Image 2 in terms of how close the agent has gotten towards achieving the goal?

        Text Responses:
            {vlm_analysis}

        
        Is the goal better achieved in Image 1 or Image 2? 
         - **Reply a single line** of 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2. 
         - Else **reply a single line** of -1 if the text is unsure or there is no difference.  
         - Limit your response to one of these numbers: -1, 0, or 1. 
         - Do not include any other details in your response.
    """

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": LABEL_TEMPLATE},
                ],
            },
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    content = response.json()["choices"][0]["message"]["content"]
    nums = re.findall(r"-?\d+", content)

    return nums[0]


if __name__ == "__main__":
    objective = "get the elf to the giftbox while avoiding the ponds"
    image1 = "/home/samrat/Documents/rl-experiments/frozen-lake-stable-baselines-rlvlmf/test/image.png"
    image2 = "/home/samrat/Documents/rl-experiments/frozen-lake-stable-baselines-rlvlmf/test/image2.png"

    with open(image1, "rb") as file:
        image1 = file.read()

    with open(image2, "rb") as file:
        image2 = file.read()

    analysis = generate_analysis(
        image1,
        image2,
        objective,
    )
    print(analysis)

    label = generate_label(analysis, objective)
    print(label)

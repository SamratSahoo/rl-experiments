from PIL import Image
import torchvision

from transformers import AutoProcessor, LlavaForConditionalGeneration  # type: ignore
import numpy as np
from dotenv import load_dotenv

load_dotenv()


model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")


VLM_POLICY_PROMPT = """
I will give you a question based on an image; you should answer in the following format.

EXAMPLE:
Q: You are the elf. Your goal is to get to the gift box. You can go left, right, up, or down. Which direction should you go in?

A: I am trying to reach the gift box in the bottom-left corner. If I go up, there is a pond trap that will result in my death, so I cannot go up. If I go left, I will stay in the same place because that is the edge of the map. I can go either right or downwards, but going right would take me further away from the gift box. Therefore, I will go downwards.

{
"direction": "down"
}

YOUR QUESTION:
Q: You are the elf. Your goal is to get to the gift box. You can go left, right, up, or down. Which direction should you go in?

A:
"""


def generate_vlm_policy(image_t, question_text=VLM_POLICY_PROMPT):
    prompt = f"USER: <image>\{question_text} ASSISTANT:"
    image = Image.from_array(np.array(image_t))
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_new_tokens=15)
    return processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


if __name__ == "__main__":
    image_t = torchvision.transforms.PILToTensor()(Image.open("./test/image.png"))

    print(
        generate_vlm_policy(
            image_t,
        )
    )

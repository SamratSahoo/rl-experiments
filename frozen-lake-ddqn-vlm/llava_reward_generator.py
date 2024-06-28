from PIL import Image
import requests
import torchvision
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")


def get_state_embedding(question_text, image_t):
    prompt = f"USER: <image>\{question_text} ASSISTANT:"
    image = Image.from_array(np.array(image_t))
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_new_tokens=15)
    return processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


if __name__ == "__main__":
    image = torchvision.transforms.PILToTensor()(Image.open("./test/image.png"))

    print(
        get_state_embedding(
            "Describe the spatial location the green player. Do not describe anything else.",
            image,
        )
    )

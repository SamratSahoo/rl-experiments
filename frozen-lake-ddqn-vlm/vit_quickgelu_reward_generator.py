import open_clip
from matplotlib import pyplot as plt
import torch
from PIL import Image
import open_clip
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14-378-quickgelu", pretrained="dfn5b"
)
model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")


def vit_infer_reward(current_state, next_state, text, iteration=0):
    text = tokenizer(text).to(device)

    with torch.inference_mode():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    current_state_image = Image.fromarray(current_state).convert("RGB")
    image_current = preprocess(current_state_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        image_features_current = model.encode_image(image_current)
        image_features_current /= image_features_current.norm(dim=-1, keepdim=True)
        similarity_current = (
            (100.0 * image_features_current @ text_features.T).cpu().numpy()[0][0]
        )

    next_state_image = Image.fromarray(next_state).convert("RGB")
    image_next = preprocess(next_state_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        image_features_next = model.encode_image(image_next)
        image_features_next /= image_features_next.norm(dim=-1, keepdim=True)
        similarity_next = (
            (100.0 * image_features_next @ text_features.T).cpu().numpy()[0][0]
        )

    similarity_reward = 1 if similarity_next - similarity_current > 0 else 0

    if iteration % 1000 == 0:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(current_state_image)
        ax[0].axis("off")
        ax[0].set_title(f"Similarity: {similarity_current:.2f}")

        ax[1].imshow(next_state_image)
        ax[1].axis("off")
        ax[1].set_title(f"Similarity: {similarity_next:.2f}")

        plt.figtext(
            0.5,
            0.01,
            f"Similarity Reward: {similarity_reward}",
            ha="center",
            fontsize=12,
        )

        output_path = f"vit-quickgelu-scores/iteration-{iteration}.png"
        plt.savefig(output_path)
        plt.close(fig)

    return similarity_reward


if __name__ == "__main__":
    current_state = np.asarray(Image.open("./test/image.png"))
    next_state = np.asarray(Image.open("./test/image2.png"))

    print(
        vit_infer_reward(
            current_state, next_state, "The elf should be on top of the gift box"
        )
    )

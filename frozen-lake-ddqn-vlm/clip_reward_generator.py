# Based on: https://github.com/yufeiwang63/RL-VLM-F/blob/main/vlms/clip_infer.py
import clip
from matplotlib import pyplot as plt
import torch
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-L/14@336px", device=device)


def clip_infer_reward(current_state, next_state, text, iteration=0):
    text = clip.tokenize(text).to(device)

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

    similarity_difference = similarity_next - similarity_current

    if similarity_difference == 0:
        similarity_reward = -1
    elif similarity_difference > 0:
        similarity_reward = similarity_difference
    else:
        similarity_reward = 0

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

        output_path = f"vlm-scores/iteration-{iteration}.png"
        plt.savefig(output_path)
        plt.close(fig)

    return similarity_reward

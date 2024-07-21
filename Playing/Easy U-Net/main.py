import argparse
import os

import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy import ndarray
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import Resize, ToTensor, Compose

from model.u_net import UNet


# Load the model
def load_model(path: str, model_class: Module = UNet) -> Module:
    model = model_class(3, 1)
    model.load_state_dict(
        torch.load(path, map_location=torch.device('cpu')))  # Load on CPU; change to 'cuda' if GPU is available
    model.eval()
    return model


# Load and preprocess the image
def load_image(path: str) -> Tensor:
    image = Image.open(path).convert('RGB')
    transform = Compose([
        Resize((512, 512)),
        ToTensor()  # Example normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# Make prediction
def predict_mask(model: Module, image_input_tensor: Tensor, threshold=0.001) -> Tensor:
    with torch.no_grad():
        prediction = model(image_input_tensor)
    return torch.sigmoid(prediction) > threshold  # Apply sigmoid and threshold for binary mask


# Visualize results
def visualize_single_image(image_path: str, model: Module):
    image_tensor = load_image(image_path)
    mask_tensor = predict_mask(model, image_tensor)
    masked_image_tensor = image_tensor * mask_tensor.float()

    ax: ndarray[Axes]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image_tensor.squeeze().permute(1, 2, 0).numpy())
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(mask_tensor.squeeze().numpy(), cmap='gray')
    ax[1].set_title('Predicted Mask')
    ax[1].axis('off')

    ax[2].imshow(masked_image_tensor.squeeze().permute(1, 2, 0).numpy())
    ax[2].set_title('Masked Image')
    ax[2].axis('off')

    plt.show()


def visualize_images_and_masks(image_paths: list[str], model: Module):
    # Determine the number of rows (one per image, maximum of 4 images)
    rows = min(len(image_paths), 4)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 5 * rows))

    if rows == 1:
        axes = [axes]  # Make it a list for uniform handling

    for ax, image_path in zip(axes, image_paths[:rows]):
        image_tensor = load_image(image_path)
        mask_tensor = predict_mask(model, image_tensor)
        masked_image_tensor = image_tensor * mask_tensor.float()

        ax[0].imshow(image_tensor.squeeze().permute(1, 2, 0).numpy())
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(masked_image_tensor.squeeze().permute(1, 2, 0).numpy())
        ax[1].set_title('Masked Image')
        ax[1].axis('off')

    plt.tight_layout()
    plt.show()


def main(image_input=None):
    model_path = "./easy_unet_model.pth"
    loaded_model = load_model(model_path)

    if image_input is None:
        # Default image if none is provided
        image_input = ["checking car 2.jpg"]
    elif image_input in ['.', './']:
        # Current directory images
        image_input = [os.path.join(os.getcwd(), f) for f in os.listdir(os.getcwd()) if
                       f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    elif os.path.isdir(image_input):
        # Given directory images
        image_input = [os.path.join(image_input, f) for f in os.listdir(image_input) if
                       f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if len(image_input) == 1:
        visualize_single_image(image_input[0], loaded_model)
    else:
        visualize_images_and_masks(image_input, loaded_model)


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process or visualize images with a U-Net model.')
    parser.add_argument('input', nargs='?', default=None, help='Path to an image, "." or a directory path.')
    args = parser.parse_args()
    main(args.input)

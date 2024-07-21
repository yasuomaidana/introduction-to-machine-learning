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
    image = Image.open(path)
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
def visualize_results(original: Tensor, mask: Tensor, masked_image: Tensor):
    ax: ndarray[Axes]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(original.squeeze().permute(1, 2, 0).numpy())  # Remove batch dim and convert to numpy
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(mask[0].squeeze().numpy(), cmap='gray')  # Squeeze to remove batch and channel dims
    ax[1].set_title('Predicted Mask')
    ax[1].axis('off')

    ax[2].imshow(masked_image.squeeze().permute(1, 2, 0).numpy())
    ax[2].set_title('Masked Image')
    ax[2].axis('off')

    plt.show()


# Main execution
if __name__ == "__main__":
    model_path = "./easy_unet_model.pth"
    image_path = "checking car 2.jpg"

    # Load the model
    loaded_model = load_model(model_path)

    # Load and prepare the image
    image_tensor = load_image(image_path)

    # Predict the mask
    predicted_mask = predict_mask(loaded_model, image_tensor)

    # Apply the mask to the original image
    masked_image_result = image_tensor * predicted_mask.float()  # Float for multiplication

    # Visualize
    visualize_results(image_tensor, predicted_mask, masked_image_result)

import torch
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Compose
from matplotlib import pyplot as plt

from model.u_net import UNet


def print_hi(name):
    model_path = "easy_unet_model.pth"
    model = UNet(3, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Transforms for the input image
    transform = Compose([
        Resize((512, 512)),
        ToTensor(),
    ])

    # Load an image for prediction
    input_image_path = 'checking car 2.jpg'
    img = Image.open(input_image_path)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply transform and make prediction
    input_tensor = transform(img).float().to(device).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)

    # Post-processing: Convert back to numpy and apply thresholding (optional)
    predicted_mask = output.squeeze(0).cpu().detach().permute(1, 2, 0)
    print(predicted_mask)
    print(predicted_mask.max(), predicted_mask.min())
    predicted_mask[predicted_mask < 0] = 0
    predicted_mask[predicted_mask > 0] = 1


    fig, axs = plt.subplots(1, 2, figsize=(20, 7))
    # Display Input Image
    axs[0].set_title("Input Image")
    axs[0].imshow(input_tensor.squeeze(0).permute(1, 2, 0))

    # Display Predicted Mask
    axs[1].set_title("Predicted Mask")
    axs[1].imshow(predicted_mask)

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

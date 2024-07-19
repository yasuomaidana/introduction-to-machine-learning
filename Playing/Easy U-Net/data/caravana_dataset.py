import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor


class CaravanDataset(Dataset):
    def __init__(self, root_dir, directory_name="train"):
        self.root_dir = f"{root_dir}/{directory_name}"
        images_directory = f"{root_dir}/{directory_name}_images"
        masks_directory = f"{root_dir}/{directory_name}_masks"

        self.images = sorted([f"{images_directory}/{img}" for img in os.listdir(images_directory)])
        self.masks = sorted([f"{masks_directory}/{mask}" for mask in os.listdir(masks_directory)])

        self.transform = Compose([Resize((512, 512)), ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = CaravanDataset("./raw")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(dataset[0][0].permute(1, 2, 0))
    axs[1].imshow(dataset[0][1].squeeze(), cmap="gray")
    plt.show()


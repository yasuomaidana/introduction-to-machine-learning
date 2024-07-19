import os

from PIL import Image
from numpy.random import shuffle
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor


class CaravanDataset(Dataset):
    def __init__(self, root_dir, directory_name="train", ratio=0.9):
        self.root_dir = f"{root_dir}/{directory_name}"
        images_directory = f"{root_dir}/{directory_name}_images"
        masks_directory = f"{root_dir}/{directory_name}_masks"
        images = sorted([f"{images_directory}/{img}" for img in os.listdir(images_directory)])
        masks = sorted([f"{masks_directory}/{mask}" for mask in os.listdir(masks_directory)])

        data = list(zip(images, masks))
        shuffle(data)

        self.eval_mode = False
        split_index = int(ratio * len(data))

        self.training_images = [img for img, _ in data[:split_index]]
        self.training_masks = [mask for _, mask in data[:split_index]]
        self.validation_images = [img for img, _ in data[split_index:]]
        self.validation_masks = [mask for _, mask in data[split_index:]]

        self.transform = Compose([Resize((512, 512)), ToTensor()])

    def __getitem__(self, index):
        if self.eval_mode:
            img, mask = self.validation_images[index], self.validation_masks[index]
        else:
            img, mask = self.training_images[index], self.training_masks[index]
        img = Image.open(img).convert("RGB")
        mask = Image.open(mask).convert("L")
        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.training_images) if not self.eval_mode else len(self.validation_images)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = CaravanDataset("./raw")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(dataset[0][0].permute(1, 2, 0))
    axs[1].imshow(dataset[0][1].squeeze(), cmap="gray")
    plt.show()


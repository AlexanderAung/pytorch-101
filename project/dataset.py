import imageio.v3 as iio
from pathlib import Path
import torch
from torchvision import datasets, transforms

img_folder_path = Path(__file__).parents[1] / "traffic_signs" / "Train"


# img_arr = iio.imread(img_path)
# img_t = torch.from_numpy(img_arr)
# img_t = img_t.permute(2, 0, 1)
# def load_dataset(root_path):
#    return datasets.ImageFolder(root_path)
dataset = datasets.ImageFolder(root=img_folder_path, transform=transforms.ToTensor())
print(dataset)

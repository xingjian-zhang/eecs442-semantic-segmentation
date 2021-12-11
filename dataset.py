import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from config import IMAGE_HEIGHT, IMAGE_WIDTH


class CarvanaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = self.init_transform(transform)
        self.images = self.df['car_path'].tolist()
        self.masks = self.df['mask_path'].tolist()
        assert len(self.images) == len(self.masks), 'number of images data not equal to masks data'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

    def init_transform(self, transform):
        if transform is None:
            return None
        if transform == "train":
            return A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    A.Rotate(limit=35, p=1.0),
                    A.HorizontalFlip(p=.1),
                    A.VerticalFlip(p=.5),
                    A.Normalize(
                        mean=[0., 0., 0.],
                        std=[1., 1., 1.],
                        max_pixel_value=255.,
                    ),
                    ToTensorV2(),
                ]
            )
        elif transform == "test" or transform == "val":
            return A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    A.Normalize(
                        mean=[0., 0., 0.],
                        std=[1., 1., 1.],
                        max_pixel_value=255.,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            raise NotImplementedError("Not supported transform {}".format(transform))

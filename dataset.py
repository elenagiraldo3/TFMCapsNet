from torch.utils.data import Dataset
import pandas as pd
import torch
from skimage import io
from PIL import Image, ImageFile
from tqdm import tqdm
import os


class DumpstersDataset(Dataset):
    """Custom Dumpsters Dataset"""

    def __init__(self, csv_file, columns, transform=None):
        """

        Args:
            csv_file (string): Path to the csv file with annotations.
        """

        dataframe = pd.read_csv(csv_file, skipinitialspace=True, usecols=columns)
        images_path = dataframe[dataframe.columns[0]]
        images = []
        self.transform = transform
        self.labels = dataframe.drop(dataframe.columns[0], axis=1)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        pbar = tqdm(total=len(images_path.values), leave=False, position=0)
        pbar.set_description("Loading images from " + csv_file)
        for idx, image_path in enumerate(images_path.values):
            if os.path.exists(image_path):
                image = io.imread(image_path, plugin='matplotlib')
                if len(image.shape) > 3:
                    image = image[0]
                image = Image.fromarray(image)
                images.append(image)
            else:
                self.labels = self.labels.drop([idx])

            pbar.update()
        pbar.close()
        self.images = pd.Series(images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images.iloc[idx]
        if self.transform:
            image = self.transform(image)
        labels = self.labels.iloc[idx]
        labels = torch.tensor(labels.values)

        return image, labels

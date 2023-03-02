from torch.utils.data import Dataset
# from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

import torch


class OTU_2dDataSet(Dataset):
    def __init__(self, images_path, labels_path, data_csv):
        self.images_path = images_path
        self.labels_path = labels_path
        self.dataDf = pd.read_csv(data_csv, delimiter='  ',  index_col=False)
        self.transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor()
                    ])
        
    def __len__(self):
        return len(self.dataDf)
    
    
    def __getitem__(self, idx):
        item = self.dataDf.iloc[idx]
        im_path = str(self.images_path/item.file)
        lbl_path = str(self.labels_path/item.file).replace('.JPG', '_binary_binary.PNG')

        im = self.transform(Image.open(im_path))/255
        lbl = self.transform(Image.open(lbl_path)).float()

        # im =  torch.rand(3, 256, 256)
        # lbl = torch.randint(0, 2, size=(1, 256, 256), dtype=torch.float)
        return im, lbl
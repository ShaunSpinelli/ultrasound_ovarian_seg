from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd


class OTU_2dDataSet(Dataset):
    def __init__(self, images_path, labels_path, data_csv):
        self.images_path = images_path
        self.labels_path = labels_path
        self.dataDf = pd.read_csv(data_csv, delimiter='  ',  index_col=False)
        
    def __len__(self):
        return len(self.dataDf)
    
    
    def __getitem__(self, idx):
        item = self.dataDf.iloc[idx]
        im_path = str(self.images_path/item.file)
        lbl_path = str(self.labels_path/item.file).replace('.JPG', '_binary_binary.PNG')
        im = read_image(im_path)/255
        lbl = (read_image(lbl_path)[0,:,:]) > 0 # convert to bool

        return im, lbl
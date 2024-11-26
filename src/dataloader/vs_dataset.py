import torch
from torch.utils.data import Dataset, DataLoader


class VideoSummarizationDataset(Dataset):
    def __init__(self, data_path):
        super(VideoSummarizationDataset, self).__init__()
        self.data_path = data_path
        self.data = self._load_data()

    def _load_data(self):
        """
        Load data from `self.data_path`.
        """
    
    def _prepare_data(self):
        """
        Prepare data for inference.
        """

def extract_image_features_and_save_them(image_path_list):
    """
    提取图像特征并保存到文件中
    """
    pass
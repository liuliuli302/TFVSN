import json
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class VideoSummarizationDataset(Dataset):
    def __init__(
            self,
            root_path="./data", 
            dataset_name="SumMe",
        ):
        super(VideoSummarizationDataset, self).__init__()
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.dataset_dir = Path(root_path, dataset_name)

        self.data = self._load_data()
        self._prepare_data()

    def _load_data(self):
        """
        Load data from `self.data_path`.
        """
        # 1 Load hdf file to dict.
        dataset_name_lower = self.dataset_name.lower()
        hdf_file_path = Path(self.dataset_dir, f"{dataset_name_lower}.h5")
        hdf_file = h5py.File(hdf_file_path, "r")

        data_part_1 = hdf5_to_dict(hdf_file)
        
        # 2 Load video_name dict.
        video_name_dict_file_path = Path(self.dataset_dir, "video_name_dict.json")
        with open(video_name_dict_file_path, "r") as f:
            video_name_dict = json.load(f)
        
        

    def _prepare_data(self):
        """
        Prepare data for inference.
        """

def hdf5_to_dict(hdf5_file):
    def recursively_convert(h5_obj):
        if isinstance(h5_obj, h5py.Group):
            return {key: recursively_convert(h5_obj[key]) for key in h5_obj.keys()}
        elif isinstance(h5_obj, h5py.Dataset):
            return h5_obj[()]
        else:
            raise TypeError("Unsupported h5py object type")
    return recursively_convert(hdf5_file)



def extract_image_features_and_save_them(image_path_list):
    """
    提取图像特征并保存到文件中
    """
    pass


if __name__ == "__main__":
    dataset = VideoSummarizationDataset()
    print("Done.")

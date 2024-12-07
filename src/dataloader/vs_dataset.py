import json
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
from PIL import Image


class VideoSummarizationDataset(Dataset):
    def __init__(
        self,
        root_path="./data",
        dataset_name="SumMe",
    ):
        super().__init__()
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.dataset_dir = Path(root_path, dataset_name)
        self.frames_dir = Path(self.dataset_dir, "frames")

        self.data_list, self.video_name_dict = self._load_data()

        # Invert the values and keys in the self.video_name_dict
        self.video_name_dict_inv = {v: k for k, v in self.video_name_dict.items()}
        # pprint(self.video_name_dict_inv)

    def _load_prompt(self):
        # 加载提示文本
        context_prompt = "If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
        format_prompt = "Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with ']'."
        summary_prompt = "Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."
        
        
    def __len__(self):
        return len(self.video_name_dict)

    def __getitem__(self, idx):
        video_name_idx = f"video_{idx + 1}"
        video_name_real = self.video_name_dict_inv[video_name_idx]
        video_frames_dir = Path(self.frames_dir, video_name_real)

        video_info = self.data_list[video_name_idx]

        picks = video_info["picks"]
        keys = list(video_info.keys())
        # Convert picks to 6-digit integer.
        picks = [f"{pick:06d}" for pick in picks]
        # Gets all file names from picks.
        frame_file_paths = [Path(video_frames_dir, f"{pick}.jpg") for pick in picks]

        video_info["frame_file_paths"] = frame_file_paths

        # Debug info.
        # pprint(frame_file_paths)
        # pprint(keys)

        return video_info

    def _load_data(self):
        """
        Load data from `self.data_path`.
        """
        # 1 Load hdf file to dict.
        dataset_name_lower = self.dataset_name.lower()
        hdf_file_path = Path(self.dataset_dir, f"{dataset_name_lower}.h5")
        hdf_file = h5py.File(hdf_file_path, "r")

        hdf_dict = hdf5_to_dict(hdf_file)
        video_names = list(hdf_dict.keys())
        keys = list(hdf_dict["video_1"].keys())

        # 2 Load video_name dict.
        video_name_dict_file_path = Path(self.dataset_dir, "video_name_dict.json")
        with open(video_name_dict_file_path, "r") as f:
            video_name_dict = json.load(f)

        return hdf_dict, video_name_dict


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
    sample0 = dataset[0]
    keys = list(sample0.keys())
    pprint(sample0["frame_file_paths"])
    pprint(keys)

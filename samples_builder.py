import os
from pathlib import Path
from pprint import pprint
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
from tqdm import tqdm

class SamplesBuilder(Dataset):
    """适用于VideoSummarizationPipeline的样本构建器，其本质是一个torch语境下的Dataset
    一个sample包含以下字段:
    - images: list, 帧文件的路径
    - prompt: str, 大模型提示词
    - video_time: float, 视频总时长(秒)
    - frame_time: str, 帧时间点，逗号分隔的字符串
    - picks: list, 所选的帧号
    - video_name: str, 视频文件名
    - dataset_name: str, 该视频来源的数据集名称
    - mode: str, "turn" or "jump"
    """

    def __init__(
            self, 
            data_dir, 
            prompt=None, 
            clip_length=5, 
            mode="both"
        ):
        """
        Args:
            data_dir (str): 数据集根目录路径
            prompt (str, optional): 大模型提示词，如果为None则使用默认提示词
            clip_length (int, optional): 每个片段的帧数，默认为5
            mode (str, optional): 选择样本模式，"turn"/"jump"/"both"
        """
        self.data_dir = Path(data_dir)
        self.clip_length = clip_length
        self.mode = mode
        self.samples = []
        
        # 默认提示词
        self.prompt = prompt or "You are a professional short film editor and director. Please score the frames divided based on theirs representativeness, diversity, and interest on a scale from 0 to 1. You may need to refer to the context for rating. And give the final score list like `[scores]`.\n without any extra text. You must output score."
        
        # 构建数据集
        self._build_dataset()

    def _build_dataset(self):
        """构建数据集，将所有样本加载到self.samples中"""
        # 加载两个数据集的基本信息
        datasets = {
            "SumMe": {
                "h5_path": Path(self.data_dir, "SumMe", "summe.h5"),
                "json_path": Path(self.data_dir, "SumMe", "video_name_dict.json"),
                "frame_dir": Path(self.data_dir, "SumMe", "frames"),
                "video_dir": Path(self.data_dir, "SumMe", "videos"),
            },
            "TVSum": {
                "h5_path": Path(self.data_dir, "TVSum", "tvsum.h5"),
                "json_path": Path(self.data_dir, "TVSum", "video_name_dict.json"),
                "frame_dir": Path(self.data_dir, "TVSum", "frames"),
                "video_dir": Path(self.data_dir, "TVSum", "videos"),
            }
        }
        
        # 处理每个数据集，添加进度条
        for dataset_name, paths in tqdm(datasets.items(), desc="Processing datasets", unit="dataset"):
            self._process_dataset(dataset_name, paths)
            
    def _process_dataset(self, dataset_name, paths):
        """处理单个数据集(SumMe或TVSum)"""
        # 加载h5文件和视频名称映射
        dataset_dict = self._hdf5_to_dict(paths["h5_path"])
        name_dict = self._load_json(paths["json_path"])
        name_dict_reverse = {v: k for k, v in name_dict.items()}
        
        # 处理每个视频，添加进度条
        for video_name in tqdm(dataset_dict.keys(), desc=f"Processing videos in {dataset_name}", unit="video"):
            video_name_real = name_dict_reverse[video_name]
            frames_dir = Path(paths["frame_dir"], video_name_real)
            video_path = Path(paths["video_dir"], f"{video_name_real}.mp4")
            
            # 获取视频信息
            video_info = self._get_video_info(video_path)
            video_dict = dataset_dict[video_name]
            picks = video_dict["picks"]
            
            # 根据模式选择处理方法
            if self.mode in ["turn", "both"]:
                self._add_clips(dataset_name, video_name, video_name_real, frames_dir, 
                               picks, video_info, "turn")
            
            if self.mode in ["jump", "both"]:
                self._add_clips(dataset_name, video_name, video_name_real, frames_dir, 
                               picks, video_info, "jump")

    def _add_clips(self, dataset_name, video_name, video_name_real, frames_dir, 
                  picks, video_info, clip_type):
        """添加指定类型的片段到样本集"""
        # 根据类型获取片段
        if clip_type == "turn":
            clips, remainder = self._get_clips_turn(picks, self.clip_length)
        else:
            clips, remainder = self._get_clips_jump(picks, self.clip_length)
            
        # 处理每个片段，添加进度条
        for i, clip in enumerate(tqdm(clips, desc=f"Building {clip_type} clips for {video_name}", unit="clip", leave=False)):
            # 计算帧时间
            frame_time = [frame / video_info["fps"] for frame in clip]
            frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
            
            # 构建样本
            sample = {
                "images": [str(Path(frames_dir, f"{str(frame).zfill(6)}.jpg")) for frame in clip],
                "prompt": self.prompt,
                "video_time": video_info["duration"],
                "frame_time": frame_time_str,
                "picks": clip,
                "video_name": video_name,
                "dataset_name": dataset_name,
                "mode": clip_type
            }
            self.samples.append(sample)

    def _get_video_info(self, video_path):
        """获取视频信息，包括时长和帧率"""
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps())
        return {"duration": video_time, "fps": fps}

    def _hdf5_to_dict(self, hdf5_file):
        """将h5文件读取为dict"""
        hdf5_file = h5py.File(hdf5_file, "r")
        
        def recursively_convert(h5_obj):
            if isinstance(h5_obj, h5py.Group):
                return {key: recursively_convert(h5_obj[key]) for key in h5_obj.keys()}
            elif isinstance(h5_obj, h5py.Dataset):
                return h5_obj[()]
            else:
                raise TypeError("Unsupported h5py object type")
                
        return recursively_convert(hdf5_file)
        
    def _load_json(self, json_path):
        """加载JSON文件"""
        import json
        with open(json_path, "r") as f:
            return json.load(f)

    def _get_clips_turn(self, picks, clip_length=5):
        """每clip_length帧划分为1个clip，不足5帧的clip忽略"""
        clips = []
        reminder = len(picks) % clip_length
        n = len(picks) - reminder
        for i in range(0, n, clip_length):
            clips.append(picks[i:i+clip_length])
        return clips, reminder

    def _get_clips_jump(self, picks, num_seg=5):
        """将picks划分为n段，之后针对每段跳帧取clip"""
        num_samples = len(picks) // num_seg
        reminder = len(picks) % num_samples
        clips = []
        for i in range(num_samples):
            indices = []
            for j in range(num_seg):
                indices.append(i+j*num_samples)
            clips.append([picks[idx] for idx in indices])
        return clips, reminder



    def __len__(self):
        """返回样本数量"""
        return len(self.samples)

    def __getitem__(self, idx):
        """返回指定索引的样本"""
        return self.samples[idx]
        
    def __call__(self, data_dir, clip_length=5, mode="both"):
        """使类实例可调用，返回自身实例
        Args:
            data_dir: 数据目录
            clip_length: 剪辑长度
            mode: 模式
        """
        print(f"Building samples from {data_dir} (mode: {mode}, clip_length: {clip_length})...")
        return self
    

# 测试代码
if __name__ == "__main__":
    data_dir = "/root/autodl-tmp/data/"
    dataset = SamplesBuilder(data_dir, clip_length=5, mode="both")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    pprint(dataset[0])
    print(len(dataset))

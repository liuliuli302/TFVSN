import numpy as np
import h5py
import json
import re
import os
from pathlib import Path
import copy


class DataCleaner:
    """用于清理和处理LLM输出数据的类"""
    
    def __init__(self, frame_interval: int = 15):
        """
        Args:
            frame_interval: 帧间隔,用于计算score_index
        """
        self.frame_interval = frame_interval
        self.score_pattern = r"Score:\s*(\d+\.\d+)"
    
    @staticmethod
    def hdf5_to_dict(hdf5_file: str) -> dict:
        """将HDF5文件转换为字典
        Args:
            hdf5_file: HDF5文件路径
        Returns:
            dict: 转换后的字典
        """
        def recursively_convert_to_dict(h5_obj):
            if isinstance(h5_obj, h5py.Dataset):
                return h5_obj[()]
            elif isinstance(h5_obj, h5py.Group):
                return {
                    key: recursively_convert_to_dict(h5_obj[key]) 
                    for key in h5_obj.keys()
                }
            else:
                raise TypeError(f"Unsupported type: {type(h5_obj)}")

        with h5py.File(hdf5_file, "r") as f:
            return recursively_convert_to_dict(f)

    def get_score_index_from_image_list(self, image_list: list) -> list:
        """从图像路径列表中获取分数索引
        Args:
            image_list: 图像路径列表
        Returns:
            list: 分数索引列表
        """
        score_index = []
        for path in image_list:
            index = int(int(os.path.basename(path).split(".")[0]) / self.frame_interval)
            score_index.append(index)
        return score_index

    def initialize_score_lists(self, data_dict_list: dict) -> dict:
        """初始化分数列表
        Args:
            data_dict_list: 数据字典列表,包含summe和tvsum的数据
        Returns:
            dict: 初始化的分数列表
        """
        score_list = {"summe": {}, "tvsum": {}}
        for dataset_name in data_dict_list.keys():
            for video_name in data_dict_list[dataset_name].keys():
                data_dict = data_dict_list[dataset_name]
                scores = np.zeros(int(data_dict[video_name]["n_steps"]))
                score_list[dataset_name][video_name] = scores
        return score_list

    def extract_scores(self, llm_output: str) -> np.ndarray:
        """从LLM输出中提取分数
        Args:
            llm_output: LLM的输出文本
        Returns:
            np.ndarray: 提取的分数数组
        """
        scores = re.findall(self.score_pattern, llm_output)
        scores = [float(score) for score in scores]
        return np.array(scores)

    def process_scores(self, scores: np.ndarray, index_length: int) -> np.ndarray:
        """处理分数数组,确保其长度与索引匹配
        Args:
            scores: 原始分数数组
            index_length: 需要的索引长度
        Returns:
            np.ndarray: 处理后的分数数组
        """
        if len(scores) < index_length:
            # 补齐0
            scores = np.concatenate((scores, np.zeros(index_length - len(scores))))
        elif len(scores) > index_length:
            # 截断多余的分数
            scores = scores[:index_length]
        return scores

    def clean_llm_outputs(self, result_dir: str, dataset_dir: str, scores_dir: str):
        """清理LLM输出并生成最终的分数列表
        Args:
            result_dir: LLM输出结果目录
            dataset_dir: 原始数据集目录
            scores_dir: 分数保存目录
        """
        result_dir = Path(result_dir)
        dataset_dir = Path(dataset_dir)

        # 加载结果文件
        result_file_list = [
            result_dir / "summe_dataset_jump_result.json",
            result_dir / "summe_dataset_turn_result.json",
            result_dir / "tvsum_dataset_jump_result.json",
            result_dir / "tvsum_dataset_turn_result.json",
        ]

        # 加载数据集
        summe_dict = self.hdf5_to_dict(str(dataset_dir / "SumMe" / "summe.h5"))
        tvsum_dict = self.hdf5_to_dict(str(dataset_dir / "TVSum" / "tvsum.h5"))
        data_dict_list = {"summe": summe_dict, "tvsum": tvsum_dict}

        # 初始化分数列表
        score_list = self.initialize_score_lists(data_dict_list)
        result_list = {}

        # 处理每个结果文件
        for file in result_file_list:
            with open(file, "r") as f:
                result = json.load(f)
                
            file_name = os.path.basename(file).split(".")[0]
            dataset_name = file_name.split("_")[0]
            check_out = []
            scores_list_tmp = copy.deepcopy(score_list[dataset_name])
            
            # 处理每个样本
            for sample in result:
                video_name = sample["id"].split("_")[1] + "_" + sample["id"].split("_")[2]
                images = sample["images"]
                index = self.get_score_index_from_image_list(images)
                index = np.array(index)
                
                # 提取和处理分数
                scores = self.extract_scores(sample["llm_out"])
                scores = self.process_scores(scores, len(index))
                
                # 检查分数长度是否匹配
                check_out.append(1 if len(scores) == len(index) else 0)
                
                # 写入分数列表
                scores_list_tmp[video_name][index] = scores
            
            # 转换分数为列表以便JSON序列化
            for video_name in scores_list_tmp:
                scores_list_tmp[video_name] = scores_list_tmp[video_name].tolist()
            
            # 统计处理结果
            check_out = np.array(check_out)
            right = check_out.sum()
            length = len(check_out)
            print(f"{file_name} - Success rate: {right}/{length}")
            
            result_list[file_name] = scores_list_tmp

        # 保存分数列表
        scores_dir = Path(scores_dir)
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores_json = scores_dir / "raw_llm_out_scores.json"
        
        with open(scores_json, "w") as f:
            json.dump(result_list, f, separators=(",", ": "), indent=4)

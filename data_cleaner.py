import numpy as np
import h5py
import json
import re
import os
from pathlib import Path
import copy
from tqdm import tqdm


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
                try:
                    data_dict = data_dict_list[dataset_name]
                    n_steps = int(data_dict[video_name]["n_steps"])
                    scores = np.zeros(n_steps, dtype=np.float32)
                    score_list[dataset_name][video_name] = scores
                except Exception as e:
                    print(f"初始化 {dataset_name}/{video_name} 的分数列表时出错: {str(e)}")
                    # 使用列表作为回退选项，更易于JSON序列化
                    n_steps = int(data_dict[video_name]["n_steps"])
                    scores = [0.0] * n_steps
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
        try:
            # 确保scores是numpy数组
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores, dtype=np.float32)
                
            if len(scores) < index_length:
                # 补齐0
                scores = np.concatenate((scores, np.zeros(index_length - len(scores), dtype=np.float32)))
            elif len(scores) > index_length:
                # 截断多余的分数
                scores = scores[:index_length]
            return scores
        except Exception as e:
            print(f"处理分数数组时出错: {str(e)}")
            # 使用普通列表作为后备方案
            if isinstance(scores, np.ndarray):
                scores_list = scores.tolist()
            else:
                scores_list = list(scores)
            
            # 调整长度
            if len(scores_list) < index_length:
                scores_list.extend([0.0] * (index_length - len(scores_list)))
            elif len(scores_list) > index_length:
                scores_list = scores_list[:index_length]
                
            return np.array(scores_list, dtype=np.float32)

    def _make_json_serializable(self, obj):
        """将对象转换为可JSON序列化的形式
        
        Args:
            obj: 输入对象
            
        Returns:
            可JSON序列化的对象
        """
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()  # 将NumPy数组转换为列表
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # 其他不可序列化类型转为字符串
            return str(obj)

    def clean_llm_outputs(self, result_dir: str, dataset_dir: str, scores_dir: str):
        """清理LLM输出并生成最终的分数列表
        Args:
            result_dir: LLM输出结果目录
            dataset_dir: 原始数据集目录
            scores_dir: 分数保存目录
        Returns:
            str: 生成的分数文件路径
        """
        result_dir = Path(result_dir)
        dataset_dir = Path(dataset_dir)
        scores_dir = Path(scores_dir)
        
        # 创建分数目录和中间结果目录
        scores_dir.mkdir(parents=True, exist_ok=True)
        intermediate_dir = scores_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否有已有的完整结果
        scores_json = scores_dir / "raw_llm_out_scores.json"
        if os.path.exists(scores_json):
            print(f"发现已存在的分数文件: {scores_json}")
            return str(scores_json)
        
        # 加载结果文件
        result_file_list = [
            result_dir / "summe_dataset_jump_result.json",
            result_dir / "summe_dataset_turn_result.json",
            result_dir / "tvsum_dataset_jump_result.json",
            result_dir / "tvsum_dataset_turn_result.json",
        ]

        # 检查结果文件是否存在
        missing_files = [str(f) for f in result_file_list if not f.exists()]
        if missing_files:
            print(f"警告: 以下结果文件不存在: {missing_files}")
            result_file_list = [f for f in result_file_list if f.exists()]
        
        # 加载数据集
        summe_dict = self.hdf5_to_dict(str(dataset_dir / "SumMe" / "summe.h5"))
        tvsum_dict = self.hdf5_to_dict(str(dataset_dir / "TVSum" / "tvsum.h5"))
        data_dict_list = {"summe": summe_dict, "tvsum": tvsum_dict}

        # 初始化分数列表
        score_list = self.initialize_score_lists(data_dict_list)
        result_list = {}        # 检查中间结果
        processed_files = {}
        for file_path in intermediate_dir.glob("*.json"):
            file_name = file_path.stem
            try:
                with open(file_path, "r") as f:
                    processed_files[file_name] = json.load(f)
                    print(f"已加载中间结果: {file_name}")
            except json.JSONDecodeError:
                print(f"无法解析中间结果文件: {file_path}")
        
        # 处理每个结果文件
        for file in result_file_list:
            file_name = os.path.basename(file).split(".")[0]
            
            # 如果已经处理过该文件，跳过处理
            if file_name in processed_files:
                print(f"使用已有的中间结果: {file_name}")
                result_list[file_name] = processed_files[file_name]
                continue
                
            # 加载原始文件
            with open(file, "r") as f:
                result = json.load(f)
                
            dataset_name = file_name.split("_")[0]
            check_out = []
            scores_list_tmp = copy.deepcopy(score_list[dataset_name])
            
            # 处理每个样本，添加进度条
            for sample in tqdm(result, desc=f"处理 {file_name}", unit="sample"):
                try:
                    video_name = sample["id"].split("_")[1] + "_" + sample["id"].split("_")[2]
                    images = sample["images"]
                    index = self.get_score_index_from_image_list(images)
                    index = np.array(index)
                    
                    # 提取和处理分数
                    scores = self.extract_scores(sample["llm_out"])
                    scores = self.process_scores(scores, len(index))
                    
                    # 检查分数长度是否匹配
                    check_out.append(1 if len(scores) == len(index) else 0)
                    
                    # 写入分数列表 - 确保正确处理NumPy数组
                    try:
                        scores_list_tmp[video_name][index] = scores
                    except Exception as e:
                        print(f"将分数写入列表时出错: {str(e)}")
                        # 尝试转换NumPy数组为列表后再写入
                        if isinstance(scores, np.ndarray):
                            scores = scores.tolist()
                        if isinstance(index, np.ndarray):
                            index = index.tolist()
                        # 遍历方式进行赋值
                        for i, idx in enumerate(index):
                            if i < len(scores):
                                scores_list_tmp[video_name][idx] = scores[i]
                except Exception as e:
                    print(f"处理样本时出错: {str(e)}，继续下一个样本")
                    check_out.append(0)
            
            # 转换分数为列表以便JSON序列化
            for video_name in scores_list_tmp:
                scores_list_tmp[video_name] = scores_list_tmp[video_name].tolist()

            # 统计处理结果
            check_out = np.array(check_out)
            right = check_out.sum()
            length = len(check_out)
            print(f"{file_name} - 成功率: {right}/{length}")
            
            # 保存中间结果 - 确保JSON可序列化
            intermediate_file = intermediate_dir / f"{file_name}.json"
            serializable_scores = self._make_json_serializable(scores_list_tmp)
            with open(intermediate_file, "w") as f:
                json.dump(serializable_scores, f, separators=(",", ": "), indent=4)
                
            result_list[file_name] = serializable_scores

        # 保存总分数列表 - 确保完全JSON可序列化
        try:
            serializable_results = self._make_json_serializable(result_list)
            with open(scores_json, "w") as f:
                json.dump(serializable_results, f, separators=(",", ": "), indent=4)
            print(f"已保存分数结果到: {scores_json}")
        except Exception as e:
            print(f"保存最终分数文件时出错: {str(e)}")
            # 尝试更基本的方式保存
            try:
                # 先尝试单独处理每个键
                final_result = {}
                for key, value in result_list.items():
                    try:
                        # 尝试单独序列化每个键
                        json_str = json.dumps(self._make_json_serializable(value))
                        final_result[key] = json.loads(json_str)
                    except Exception:
                        print(f"无法序列化键 {key}，使用字符串替代")
                        final_result[key] = f"<非JSON可序列化数据: {type(value).__name__}>"
                
                # 保存处理后的结果
                with open(scores_json, "w") as f:
                    json.dump(final_result, f, separators=(",", ": "), indent=4)
                print(f"已以替代方式保存分数结果到: {scores_json}")
            except Exception as e2:
                print(f"替代保存方式也失败: {str(e2)}")
        return str(scores_json)

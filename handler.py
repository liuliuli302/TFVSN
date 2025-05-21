from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import warnings
import torch
import copy
import re
import numpy as np
from pathlib import Path
import os
import json
from tqdm import tqdm

warnings.filterwarnings("ignore")

class LLMHandler:
    """用来包装和处理大型语言模型的类,以适配VideoSummarizationPipeline的需求"""
    def __init__(self, model_name: str = "lmms-lab/LLaVA-Video-7B-Qwen2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_map = "auto"
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.max_length = None
        self.load_model(model_name)

    def load_model(self, model_name: str):
        """加载LLM模型和相关处理器"""
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_name, 
            None, 
            "llava_qwen",
            torch_dtype="bfloat16", 
            device_map=self.device_map
        )
        self.model = self.model.eval()
        return self.model

    def load_images_from_paths(self, paths: list) -> list:
        """从路径列表加载图像"""
        images = []
        for path in paths:
            images.append(Image.open(path).convert('RGB'))
        return images

    def preprocess_images(self, images: list) -> torch.Tensor:
        """预处理图像列表"""
        processed_images = self.image_processor.preprocess(images, return_tensors="pt")[
            "pixel_values"].to(self.device).bfloat16()
        return [processed_images]

    def generate_response(self, prompt: str, images: list, time_info: dict = None) -> str:
        """生成针对图像和提示的响应
        Args:
            prompt: 提示文本
            images: 图像列表
            time_info: 包含video_time和frame_time的字典
        """
        if time_info:
            video_time = time_info.get("video_time", 0)
            frame_time = time_info.get("frame_time", "")
            time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(images[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
            question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt}"
        else:
            question = DEFAULT_IMAGE_TOKEN + f"\n{prompt}"

        # 准备对话模板
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # 生成输入ID
        input_ids = tokenizer_image_token(
            prompt_question, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        # 生成响应
        output = self.model.generate(
            input_ids,
            images=images,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=1000,
        )
        
        # 解码响应
        text_output = self.tokenizer.batch_decode(
            output, 
            skip_special_tokens=True
        )[0].strip()

        return text_output

    def analyze_frames(self, images: list, time_info: dict = None) -> str:
        """分析帧内容并生成重要性得分
        Args:
            images: 图像列表
            time_info: 包含video_time和frame_time的字典
        Returns:
            str: 包含每帧分析和得分的字符串
        """
        # 第一次询问帧数
        initial_prompt = "HOW MANY FRAMES I GIVE TO YOU?"
        self.generate_response(initial_prompt, images, time_info)
        
        # 分析帧内容和得分
        analysis_prompt = ("For every frame I gave, first analyzed the content of each frame "
                         "and then gave the importance score of the frame in the video summarization task. "
                         "Make sure you have given scores for every frame on a scale from 0 to 1. "
                         "The output format is given in the following :\n"
                         "Frame i:frame content\n Score:[score]")
        
        return self.generate_response(analysis_prompt, images, time_info)

    def get_scores_from_response(self, response: str) -> list:
        """从响应文本中提取分数
        Args:
            response: LLM的响应文本
        Returns:
            list: 提取的分数列表
        """
        pattern = r"Score:\s*(\d+\.\d+)"
        scores = re.findall(pattern, response)
        return [float(score) for score in scores]

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

    def process_batch(self, json_data: list, result_dir: str, batch_size: int = 4):
        """处理数据集中的所有样本（忽略批处理参数）
        Args:
            json_data: 数据集样本列表
            result_dir: 结果保存目录
            batch_size: 已弃用参数，保留是为了兼容性
        """
        os.makedirs(result_dir, exist_ok=True)
        
        # 创建中间结果目录
        intermediate_dir = Path(result_dir) / "intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # 检查是否有已处理的样本
        processed_samples = []
        sample_ids = set()
        if os.path.exists(intermediate_dir):
            for file in os.listdir(intermediate_dir):
                if file.endswith(".json"):
                    sample_path = Path(intermediate_dir) / file
                    try:
                        with open(sample_path, "r") as f:
                            processed_sample = json.load(f)
                            processed_samples.append(processed_sample)
                            if "id" in processed_sample:
                                sample_ids.add(processed_sample["id"])
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"警告: 无法加载 {file}: {e}")
        
        results = processed_samples.copy()
        
        # 确保每个样本都有ID字段
        for i, sample in enumerate(json_data):
            if "id" not in sample:
                # 生成唯一ID
                video_name = sample.get("video_name", f"video_{i}")
                dataset_name = sample.get("dataset_name", "unknown")
                mode = sample.get("mode", "sample")
                sample["id"] = f"{dataset_name}_{video_name}_{mode}_{i}"
        
        # 直接处理所有尚未处理的样本
        remaining_samples = [s for s in json_data if "id" not in s or s["id"] not in sample_ids]
        for sample in tqdm(remaining_samples, desc="Processing samples", unit="sample"):
            try:
                # 加载和预处理图像
                images = self.load_images_from_paths(sample["images"])
                processed_images = self.preprocess_images(images)
                
                # 准备时间信息
                time_info = {
                    "video_time": sample["video_time"],
                    "frame_time": sample["frame_time"]
                }
                
                # 分析帧并获取结果
                llm_output = self.analyze_frames(processed_images, time_info)
                
                # 保存结果
                result_sample = copy.deepcopy(sample)
                result_sample["llm_out"] = llm_output
                results.append(result_sample)
                
                # 立即保存中间结果
                # 确保文件名有效（移除不允许的字符）
                safe_id = str(sample['id']).replace('/', '_').replace('\\', '_')
                sample_file = Path(intermediate_dir) / f"{safe_id}.json"
                
                # 创建可JSON序列化的样本副本，处理NumPy数组
                json_serializable_sample = self._make_json_serializable(result_sample)
                
                with open(sample_file, "w") as f:
                    json.dump(json_serializable_sample, f, indent=4)
                    
            except Exception as e:
                print(f"处理样本 {sample['id']} 时出错: {str(e)}")
                # 记录错误但继续处理其他样本

        return results

    def process_dataset(self, dataset, result_dir: str, batch_size: int = 4):
        """处理整个数据集，并保存中间结果
        Args:
            dataset: 数据集JSON文件路径或直接的数据集对象
            result_dir: 结果保存目录
            batch_size: 已弃用参数，保留是为了兼容性
        Returns:
            result_file_path: 结果文件路径
        """
        # 确保结果目录存在
        os.makedirs(result_dir, exist_ok=True)
        
        # 如果是路径，加载数据集
        if isinstance(dataset, (str, bytes, os.PathLike)):
            with open(dataset, "r") as f:
                dataset = json.load(f)
        
        # 检查是否有之前的完整结果文件
        result_file_name = "samples_result.json"
        result_file_path = Path(result_dir) / result_file_name
        
        if os.path.exists(result_file_path):
            print(f"发现已存在的结果文件: {result_file_path}")
            try:
                with open(result_file_path, "r") as f:
                    existing_results = json.load(f)
                    
                # 检查结果是否完整
                if len(existing_results) == len(dataset):
                    print("已有完整结果文件，无需重新处理")
                    return result_file_path
                else:
                    print(f"已有结果文件不完整 ({len(existing_results)}/{len(dataset)})")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"已有结果文件损坏: {e}")
        
        # 处理数据集，支持恢复处理
        results = self.process_batch(dataset, result_dir)
        
        # 保存完整结果 - 先转换为JSON可序列化对象
        serializable_results = self._make_json_serializable(results)
        with open(result_file_path, "w") as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"已保存完整结果到: {result_file_path}")
        return result_file_path
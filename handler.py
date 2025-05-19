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

    def process_batch(self, json_data: list, result_dir: str):
        """批量处理数据集中的样本
        Args:
            json_data: 数据集样本列表
            result_dir: 结果保存目录
        """
        os.makedirs(result_dir, exist_ok=True)
        results = []

        for sample in json_data:
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

        return results

    def process_dataset(self, dataset_path: str, result_dir: str):
        """处理整个数据集
        Args:
            dataset_path: 数据集JSON文件路径
            result_dir: 结果保存目录
        """
        # 加载数据集
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        
        # 处理数据集
        results = self.process_batch(dataset, result_dir)
        
        # 保存结果
        result_file_name = Path(dataset_path).stem + "_result.json"
        result_file_path = Path(result_dir) / result_file_name
        
        with open(result_file_path, "w") as f:
            json.dump(results, f, indent=4)

        return result_file_path
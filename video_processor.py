import os
from pathlib import Path
import cv2
import numpy as np
import torch
from typing import Dict, Tuple, Union, List
from config import VideoProcessorConfig
from tqdm import tqdm


class VideoProcessor:
    '''
    视频处理工具类
    1. 视频帧抽取方法
    2. 加载一个/多个视频到内存（以Tensor/ndarray形式返回）
    '''
    def __init__(self, config: VideoProcessorConfig):
        """初始化视频处理器
        Args:
            config: 视频处理配置对象
        """
        self.config = config
        self.frame_rate = config.frame_rate
        self.resolution = config.resolution

    def load_single_video(self, video_file_path: str) -> np.ndarray:
        """加载单个视频为ndarray
        Args:
            video_file_path: 视频文件路径
        Returns:
            np.ndarray: 视频帧数组，形状为 (num_frames, height, width, channels)
        """
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found: {video_file_path}")

        cap = cv2.VideoCapture(video_file_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 调整分辨率（如果需要）
            if frame.shape[:2] != self.resolution:
                frame = cv2.resize(frame, self.resolution)
            frames.append(frame)
            
        cap.release()
        return np.array(frames)

    def load_multi_videos(self, videos_dir_path: str) -> Dict[str, np.ndarray]:
        """加载多个视频为字典，key为视频文件名，value为ndarray数据
        Args:
            videos_dir_path: 视频文件夹路径
        Returns:
            Dict[str, np.ndarray]: 视频数据字典
        """
        if not os.path.exists(videos_dir_path):
            raise FileNotFoundError(f"Videos directory not found: {videos_dir_path}")

        videos_dict = {}
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        for file_name in os.listdir(videos_dir_path):
            if any(file_name.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(videos_dir_path, file_name)
                video_name = Path(file_name).stem
                try:
                    video_data = self.load_single_video(video_path)
                    videos_dict[video_name] = video_data
                except Exception as e:
                    print(f"Error loading video {file_name}: {e}")
                    continue
                
        return videos_dict

    def extract_frames(self, videos_dir_path: str, frames_save_dir: str, 
                      sample_rate: int = 1) -> List[Tuple[str, int]]:
        """提取指定视频文件夹下的视频帧
        Args:
            videos_dir_path: 视频文件夹路径
            frames_save_dir: 帧保存文件夹路径
            sample_rate: 采样率，每隔多少帧保存一帧
        Returns:
            List[Tuple[str, int]]: 视频名称和对应的帧数列表
        """
        if not os.path.exists(videos_dir_path):
            raise FileNotFoundError(f"Videos directory not found: {videos_dir_path}")
        
        os.makedirs(frames_save_dir, exist_ok=True)
        results = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        # 处理单个视频文件
        if os.path.isfile(videos_dir_path):
            if any(videos_dir_path.lower().endswith(ext) for ext in video_extensions):
                video_name = Path(videos_dir_path).stem
                video_frames_dir = os.path.join(frames_save_dir, video_name)
                os.makedirs(video_frames_dir, exist_ok=True)
                frame_count = self._extract_frames_from_video(
                    videos_dir_path, video_frames_dir, sample_rate)
                results.append((video_name, frame_count))
        
        # 处理视频文件夹
        else:
            for file_name in os.listdir(videos_dir_path):
                if any(file_name.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(videos_dir_path, file_name)
                    video_name = Path(file_name).stem
                    video_frames_dir = os.path.join(frames_save_dir, video_name)
                    os.makedirs(video_frames_dir, exist_ok=True)
                    
                    try:
                        frame_count = self._extract_frames_from_video(
                            video_path, video_frames_dir, sample_rate)
                        results.append((video_name, frame_count))
                    except Exception as e:
                        print(f"Error extracting frames from {file_name}: {e}")
                        continue
        
        return results

    def _extract_frames_from_video(self, video_path: str, 
                                 save_dir: str, sample_rate: int = 1) -> int:
        """从单个视频文件提取帧
        Args:
            video_path: 视频文件路径
            save_dir: 保存目录
            sample_rate: 采样率
        Returns:
            int: 提取的帧数
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        saved_count = 0
        
        # 使用tqdm创建进度条
        with tqdm(total=total_frames, desc=f"Extracting frames from {Path(video_path).name}", unit="frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % sample_rate == 0:
                    # 调整分辨率（如果需要）
                    if frame.shape[:2] != self.resolution:
                        frame = cv2.resize(frame, self.resolution)
                        
                    frame_path = os.path.join(save_dir, f"{saved_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1
                    
                frame_count += 1
                pbar.update(1)
            
        cap.release()
        print(f"Extracted {saved_count} frames from {video_path} to {save_dir}")
        return saved_count

    def to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """将ndarray格式的帧转换为PyTorch tensor
        Args:
            frames: 帧数据，形状为 (num_frames, height, width, channels)
        Returns:
            torch.Tensor: 转换后的张量，形状为 (num_frames, channels, height, width)
        """
        # 转换为float32并归一化到[0,1]
        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32) / 255.0
            
        # 转换为PyTorch张量并调整维度顺序
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        return frames_tensor


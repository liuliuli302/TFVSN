import os
from pathlib import Path
import cv2
from config import *


class VideoProcessor:
    '''
    视频处理工具类
    1 视频帧抽取方法
    2 加载一个/多个视频到内存（以Tensor/ndarray形式返回）
    '''
    def __init__(self, config: VideoProcessorConfig):
        pass

    def load_single_video(self, video_file_path):
        # 加载单个视频为Tensor/ndarray
        pass

    def load_multi_videos(self, videos_dir_path):
        # 加载多个视频为Tensor/ndarray的字典，字典key为视频的文件名，字典的value为相应的数据
        pass

    def extract_frames(videos_dir_path, frames_save_dir):
        # 提取指定视频文件夹下的视频的所有视频帧到指定的保存文件夹下
        os.makedirs(video_frames_dir, exist_ok=True)

        video_name = Path(videos_dir_path).stem
        video_frames_dir = os.path.join(frames_save_dir, video_name)
        cap = cv2.VideoCapture(videos_dir_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(video_frames_dir, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        cap.release()
        print(
            f"Extracted {frame_count} frames from {videos_dir_path} to {video_frames_dir}")
        return video_name, frame_count


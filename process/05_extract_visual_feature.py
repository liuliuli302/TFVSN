from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model
import torch
from decord import VideoReader, cpu
import numpy as np
import math
import argparse
from pathlib import Path
from tqdm import tqdm

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载BLIP - 2模型和处理器，使用float32精度
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2Model.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32)
model.to(device)


@torch.no_grad()
def extract_and_save_features(video_folder, output_folder, stride=1, batch_size=16):
    video_folder = Path(video_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 获取视频文件列表
    video_files = list(video_folder.iterdir())
    video_files = [video_file for video_file in video_files if video_file.suffix in [
        '.mp4', '.avi', '.mov']]

    # 外层tqdm：显示视频处理进度
    for video_file in tqdm(video_files, desc="Processing videos", unit="video"):
        vr = VideoReader(str(video_file), ctx=cpu(0))
        frame_count = len(vr)

        # 生成按照stride采样的帧索引
        indices = list(range(0, frame_count, stride))
        frames = vr.get_batch(indices).asnumpy()

        all_features = []
        # 使用 math.ceil 计算总批次数量
        num_batches = math.ceil(len(frames) / batch_size)

        # 内层tqdm：显示每个视频内批次处理进度
        for batch_idx in tqdm(range(num_batches), desc=f"Processing batches in {video_file.name}", unit="batch", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(frames))
            batch_frames = frames[start_idx:end_idx]

            batch_images = [Image.fromarray(frame) for frame in batch_frames]
            # 处理图像，使用float32精度
            inputs = processor(images=batch_images, return_tensors="pt").to(
                device, torch.float32)

            # 使用get_qformer_features获取特征
            outputs = model.get_qformer_features(
                pixel_values=inputs.pixel_values)
            batch_features = outputs.last_hidden_state.squeeze().cpu().numpy()

            if len(batch_features.shape) == 1:
                batch_features = batch_features[np.newaxis, :]
            all_features.append(batch_features)

        # 拼接所有批次的特征
        final_features = np.concatenate(all_features, axis=0)

        # 保存特征到一个文件
        video_name = video_file.stem
        feature_file = output_folder / f'{video_name}.npy'
        np.save(feature_file, final_features)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Video frame feature extraction')
    parser.add_argument('--video_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--stride', type=int)
    parser.add_argument('--batch_size', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    extract_and_save_features(
        args.video_folder, args.output_folder, args.stride, args.batch_size)


if __name__ == "__main__":
    main()

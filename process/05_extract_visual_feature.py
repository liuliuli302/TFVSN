from PIL import Image
import torch
from transformers import Blip2Processor, Blip2Model
from decord import VideoReader, cpu
import numpy as np
import math
import argparse
from pathlib import Path
from tqdm import tqdm

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载BLIP-2模型和处理器，使用float32精度
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

    for video_file in tqdm(video_files, desc="Processing videos", unit="video"):
        vr = VideoReader(str(video_file), ctx=cpu(0))
        frame_count = len(vr)

        # 生成按照stride采样的帧索引
        indices = list(range(0, frame_count, stride))
        frames = vr.get_batch(indices).asnumpy()

        all_features = []
        num_batches = math.ceil(len(frames) / batch_size)

        for batch_idx in tqdm(range(num_batches), desc=f"Processing batches in {video_file.name}", unit="batch", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(frames))
            batch_frames = frames[start_idx:end_idx]

            batch_images = [Image.fromarray(frame) for frame in batch_frames]
            inputs = processor(images=batch_images, return_tensors="pt").to(
                device, torch.float32)

            # 替代 get_qformer_features：手动执行 vision -> qformer -> language projection
            vision_outputs = model.vision_model(
                pixel_values=inputs.pixel_values)
            image_embeds = vision_outputs[0]

            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
            query_tokens = model.query_tokens.expand(
                image_embeds.shape[0], -1, -1)

            qformer_outputs = model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True
            )
            query_output = qformer_outputs.last_hidden_state

            projected_features = model.language_projection(query_output)

            if projected_features.dtype != image_embeds.dtype:
                projected_features = projected_features.to(image_embeds.dtype)

            batch_features = projected_features.cpu().numpy()
            all_features.append(batch_features)

        final_features = np.concatenate(all_features, axis=0)

        video_name = video_file.stem
        feature_file = output_folder / f'{video_name}.npy'
        np.save(feature_file, final_features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    extract_and_save_features(
        args.video_folder, args.output_folder, args.stride, args.batch_size)


if __name__ == "__main__":
    main()

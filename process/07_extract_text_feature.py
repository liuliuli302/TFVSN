import torch
import numpy as np
from transformers import Blip2Processor, Blip2Model
from pathlib import Path
from tqdm import tqdm
import argparse

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和处理器
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2Model.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32)
model.to(device)


@torch.no_grad()
def extract_text_features(text_folder, output_folder):
    text_folder = Path(text_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    text_files = [f for f in text_folder.iterdir() if f.suffix == ".txt"]

    for text_file in tqdm(text_files, desc="Processing text files", unit="file"):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # 处理文本为 input_ids
        inputs = processor(text=[text], return_tensors="pt").to(device)

        input_ids = inputs["input_ids"]
        # 提取词嵌入
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
        text_features = inputs_embeds.squeeze(
            0).cpu().numpy()  # shape: (seq_len, embed_dim)

        # 保存
        feature_file = output_folder / f"{text_file.stem}_text.npy"
        np.save(feature_file, text_features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    extract_text_features(args.text_folder, args.output_folder)


if __name__ == "__main__":
    main()

import os
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import warnings
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


@torch.no_grad()
def caption_video(model, tokenizer, image_processor, video_path, max_frames_num, prompt_templates):
    video, frame_time, video_time = load_video(
        video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")[
        "pixel_values"].cuda().bfloat16()
    video = [video]

    captions = []
    for template in prompt_templates:
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{template}"
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        # 生成 attention_mask
        attention_mask = torch.ones_like(input_ids).cuda()

        cont = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            attention_mask=attention_mask
        )
        text_output = tokenizer.batch_decode(
            cont, skip_special_tokens=True)[0].strip()
        captions.append(text_output)

    return captions


def process_video_folder(video_folder, output_folder):
    pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_name = "llava_qwen"
    max_frames_num = 64
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,
                                                                          torch_dtype="bfloat16", device_map="auto")
    model.eval()

    prompt_templates = [
        "Summarize the main content and main events of the video in a concise and clear manner according to the order of events."
    ]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(
        video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_name in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_folder, video_name)
        captions = caption_video(
            model, tokenizer, image_processor, video_path, max_frames_num, prompt_templates)
        txt_name = os.path.splitext(video_name)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_name)
        with open(txt_path, 'w') as f:
            for caption in captions:
                f.write(caption + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Video Captioning')
    parser.add_argument('--video_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    process_video_folder(args.video_folder, args.output_folder)


if __name__ == "__main__":
    main()

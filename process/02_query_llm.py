# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
from pathlib import Path
import copy
import torch
import json
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm
import os
warnings.filterwarnings("ignore")
# 从image path list中加载image
def load_images_from_paths(paths):
    images = []
    for path in paths:
        images.append(Image.open(path).convert('RGB'))
    return images
def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

# 1 加载模型
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model = model.eval()

# 2 加载数据
dataset_dir = Path("/root/TFVSN/dataset")
json_files = [
    # Path(dataset_dir,"SumMe","summe_dataset_jump.json"),
    # Path(dataset_dir,"SumMe","summe_dataset_turn.json"),
    Path(dataset_dir,"TVSum","tvsum_dataset_jump.json"),
    Path(dataset_dir,"TVSum","tvsum_dataset_turn.json"),
]

# 3 执行query循环
result_dir = Path("/root/TFVSN/out")
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
for dataset in json_files:
    result_file_name = str(dataset).split('/')[-1].split('.')[0] + "_result.json"
    reslut_file = Path(result_dir, result_file_name)
    result = []
    # 加载json
    with open(dataset, "r") as f:
        dataset = json.load(f)
    for sample in tqdm(dataset, desc="Samples", leave=False):
        images = load_images_from_paths(sample["images"])
        video_time = sample["video_time"]
        frame_time = sample["frame_time"]
        prompt = sample["prompt"]
        prompt = "HOW MANY FRAMES I GIVE TO YOU?"
        # prompt = "Please describe the video content in detail."
        images = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        images = [images]
        
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(images[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{prompt}"
        
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        cont = model.generate(
            input_ids,
            images=images,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=500,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        
        # 更新conv
        conv.messages[1][1] = text_outputs
        question2 = "For every frame I gave, first analyzed the content of each frame and then gave the importance score of the frame in the video summarization task. Make sure you have given scores for every frame on a scale from 0 to 1. The output format is given in the following :\nFrame i:frame content\n Score:[score]"
        conv.append_message(conv.roles[0], question2)
        prompt2 = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt2, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        cont = model.generate(
            input_ids,
            images=images,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=1000,
            attention_mask=None,
        )
        text_outputs2 = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        res_sample = copy.deepcopy(sample)
        res_sample["llm_out"]=text_outputs2
        result.append(res_sample)
    # 保存结果
    with open(reslut_file, "w") as f:
        json.dump(result, f, indent=4)
    f.close()
import argparse
import torch
from pathlib import Path
import h5py
import json
import numpy as np

# 将h5文件读取为dict
def hdf5_to_dict(hdf5_file):
    hdf5_file = h5py.File(hdf5_file, "r")
    def recursively_convert(h5_obj):
        if isinstance(h5_obj, h5py.Group):
            return {key: recursively_convert(h5_obj[key]) for key in h5_obj.keys()}
        elif isinstance(h5_obj, h5py.Dataset):
            return h5_obj[()]
        else:
            raise TypeError("Unsupported h5py object type")
    return recursively_convert(hdf5_file)

# 每5帧划分为1个clip，不足5帧的clip忽略
def get_clips_turn(picks, clip_length=5):
    clips = []
    reminder = len(picks) % clip_length
    n = len(picks) - reminder
    for i in range(0, n, clip_length):
        clips.append(picks[i:i+clip_length])
    return clips, reminder

# 将picks划分为n段，之后针对每段跳帧取clip
def get_clips_jump(picks, num_seg=5):
    num_samples = len(picks) // num_seg
    reminder = len(picks) % num_samples
    clips = []
    for i in range(num_samples):
        indices = []
        for j in range(num_seg):
            indices.append(i+j*num_samples)
        clips.append(picks[indices])
    return clips, reminder

# 生成sample的id
def id_generator(dataset_name, video_name, clip_type, sample_id, remainder):
    # example: SumMe_video_1_00000000
    # 后八位数字的前两位表示clip的类型，00代表turn，01代表jump
    # 中间四位代表sample的id，最后两位代表reminder，也就是有多少帧被忽略
    # 默认sample数小于9999，reminder小于99
    if clip_type == "turn":
        clip_type = "00"
    elif clip_type == "jump":
        clip_type = "01"
    ids = dataset_name + "_" + video_name + "_" + clip_type + sample_id.zfill(4) + remainder.zfill(2)
    return ids

# 不同的大语言模型对conversation要求不同
def apply_conversation_template(llm_name, num_images, prompt):
    if llm_name == "llava-next":
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                # 重复`{"type":"text","text":f"Frame {i}"},{"type": "image"}``, num_images次
                {"type":"text","text":f"Frame 0"},
                {"type": "image"},
                {"type":"text","text":f"Frame 1"},
                {"type": "image"},
                {"type":"text","text":f"Frame 2"},
                {"type": "image"},
                {"type":"text","text":f"Frame 3"},
                {"type": "image"}    
            ]
            },
        ]
    return conversation

def main(data_dir, save_dir):
    # 1 读取基本信息
    # data_dir = "/home/insight/workspace/TFVSN/data"
    
    summe_h5_path = Path(data_dir,"SumMe","summe.h5")
    tvsum_h5_path = Path(data_dir, "TVSum", "tvsum.h5")
    summe_json_path = Path(data_dir, "SumMe", "video_name_dict.json")
    tvsum_json_path = Path(data_dir, "TVSum", "video_name_dict.json")
    summe_frame_dir = Path(data_dir, "SumMe", "frames")
    tvsum_frame_dir = Path(data_dir, "TVSum", "frames")

    summe_dict = hdf5_to_dict(summe_h5_path)
    tvsum_dict = hdf5_to_dict(tvsum_h5_path)

    with open(summe_json_path, "r") as f:
        summe_name_dict = json.load(f)
    with open(tvsum_json_path, "r") as f:
        tvsum_name_dict = json.load(f)

    # 将summe_name_dict和tvsum_name_dict反转
    summe_name_dict_revers = {v: k for k, v in summe_name_dict.items()}
    tvsum_name_dict_revers = {v: k for k, v in tvsum_name_dict.items()}

    # 2 生成dataset
    # context_prompt = "If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
    prompt = "You are a professional short film editor and director. Please score the frames divided based on theirs representativeness, diversity, and interest on a scale from 0 to 1. You may need to refer to the context for rating. And give the final score list like `[scores]`.\n without any extra text. You must output score."

    finaly_prompt = prompt

    dataset_samples_turn_summe = []
    dataset_samples_jump_summe = []
    dataset_samples_turn_tvsum = []
    dataset_samples_jump_tvsum = []

    summe_videos = summe_dict.keys()
    tvsum_videos = tvsum_dict.keys()

    clip_length = 10

    from decord import VideoReader, cpu

    # 生成SumMe的dataset
    for video_name in summe_videos:
        video_name_real = summe_name_dict_revers[video_name]
        frames_dir = Path(summe_frame_dir, video_name_real)
        
        # NEW CODE
        video_path = Path(data_dir,"SumMe","videos",f"{video_name_real}.mp4")
        vr = VideoReader(str(video_path), ctx=cpu(0),num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps())
        # NEW CODE
        
        video_dict = summe_dict[video_name]
        picks = video_dict["picks"]
        clips_turn, remainder_turn = get_clips_turn(picks, clip_length)
        clips_jump, remainder_jump = get_clips_jump(picks, clip_length)

        for i, clip in enumerate(clips_turn):
            # NEW CODE
            frame_time = [frame/fps for frame in clip]
            frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
            # NEW CODE
            sample = {}
            sample_id = id_generator("SumMe", video_name, "turn", str(i), str(remainder_turn))
            sample["id"] = sample_id
            sample["images"] = [str(Path(frames_dir, f"{str(frame).zfill(6)}.jpg")) for frame in clip]
            sample["prompt"] = finaly_prompt
            sample["video_time"] = video_time
            sample["frame_time"] = frame_time
            dataset_samples_turn_summe.append(sample)
        
        for i, clip in enumerate(clips_jump):
            # NEW CODE
            frame_time = [frame/fps for frame in clip]
            frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
            # NEW CODE
            sample = {}
            sample_id = id_generator("SumMe", video_name, "jump", str(i), str(remainder_jump))
            sample["id"] = sample_id
            sample["images"] = [str(Path(frames_dir, f"{str(frame).zfill(6)}.jpg")) for frame in clip]
            sample["prompt"] = finaly_prompt
            sample["video_time"] = video_time
            sample["frame_time"] = frame_time
            dataset_samples_jump_summe.append(sample)

    # 生成TVSum的dataset
    for video_name in tvsum_videos:
        video_name_real = tvsum_name_dict_revers[video_name]
        frames_dir = Path(tvsum_frame_dir, video_name_real)
        
        # NEW CODE
        video_path = Path(data_dir,"TVSum","videos",f"{video_name_real}.mp4")
        vr = VideoReader(str(video_path), ctx=cpu(0),num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps())
        # NEW CODE
        
        video_dict = tvsum_dict[video_name]
        picks = video_dict["picks"]

        clips_turn, remainder_turn = get_clips_turn(picks, clip_length)
        clips_jump, remainder_jump = get_clips_jump(picks, clip_length)

        for i, clip in enumerate(clips_turn):
            # NEW CODE
            frame_time = [frame/fps for frame in clip]
            frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
            # NEW CODE
            sample = {}
            sample_id = id_generator("TVSum", video_name, "turn", str(i), str(remainder_turn))
            sample["id"] = sample_id
            sample["images"] = [str(Path(frames_dir, f"{str(frame).zfill(6)}.jpg")) for frame in clip]
            sample["prompt"] = finaly_prompt
            sample["video_time"] = video_time
            sample["frame_time"] = frame_time
            dataset_samples_turn_tvsum.append(sample)
        
        for i, clip in enumerate(clips_jump):
            # NEW CODE
            frame_time = [frame/fps for frame in clip]
            frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
            # NEW CODE
            
            sample = {}
            sample_id = id_generator("TVSum", video_name, "jump", str(i), str(remainder_jump))
            sample["id"] = sample_id
            sample["images"] = [str(Path(frames_dir, f"{str(frame).zfill(6)}.jpg")) for frame in clip]
            sample["prompt"] = finaly_prompt
            sample["video_time"] = video_time
            sample["frame_time"] = frame_time
            dataset_samples_jump_tvsum.append(sample)

    # 3 保存dataset
    out_dir_summe = Path(save_dir, "SumMe")
    out_dir_tvsum = Path(save_dir, "TVSum")
    out_dir_summe.mkdir(parents=True, exist_ok=True)
    out_dir_tvsum.mkdir(parents=True, exist_ok=True)

    json_turn_summe = Path(out_dir_summe, "summe_dataset_turn.json")
    json_jump_summe = Path(out_dir_summe, "summe_dataset_jump.json")

    json_turn_tvsum = Path(out_dir_tvsum, "tvsum_dataset_turn.json")
    json_jump_tvsum = Path(out_dir_tvsum, "tvsum_dataset_jump.json")

    # 保存为json文件
    with open(json_turn_summe, "w") as f:
        json.dump(dataset_samples_turn_summe, f, indent=4)
    with open(json_jump_summe, "w") as f:
        json.dump(dataset_samples_jump_summe, f, indent=4)
    with open(json_turn_tvsum, "w") as f:
        json.dump(dataset_samples_turn_tvsum, f, indent=4)
    with open(json_jump_tvsum, "w") as f:
        json.dump(dataset_samples_jump_tvsum, f, indent=4)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory path to the videos.",
    )
    parser.add_argument(
        "--dataset_save_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_dir, args.dataset_save_dir)


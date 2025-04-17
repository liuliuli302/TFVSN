import argparse
from pathlib import Path
import json
import torch
import numpy as np
import h5py
import copy
import os
import re


def get_score_index_from_image_list(image_list, frame_interval):
    score_index = []
    for path in image_list:
        index = int(int(os.path.basename(path).split(".")[0]) / frame_interval)
        score_index.append(index)
    return score_index


def hdf5_to_dict(hdf5_file):
    def recursively_convert_to_dict(h5_obj):
        if isinstance(h5_obj, h5py.Dataset):
            return h5_obj[()]
        elif isinstance(h5_obj, h5py.Group):
            return {
                key: recursively_convert_to_dict(h5_obj[key]) for key in h5_obj.keys()
            }
        else:
            raise TypeError(f"Unsupported type: {type(h5_obj)}")

    with h5py.File(hdf5_file, "r") as f:
        return recursively_convert_to_dict(f)


def main(result_dir, dataset_dir, scores_dir):

    result_dir = Path(result_dir)
    dataset_dir = Path(dataset_dir)

    # 1 加载数据
    result_file_list = [
        result_dir / "summe_dataset_jump_result.json",
        result_dir / "summe_dataset_turn_result.json",
        result_dir / "tvsum_dataset_jump_result.json",
        result_dir / "tvsum_dataset_turn_result.json",
    ]

    summe_hdf = Path(dataset_dir, "SumMe", "summe.h5")
    tvsum_hdf = Path(dataset_dir, "TVSum", "tvsum.h5")

    summe_dict = hdf5_to_dict(summe_hdf)
    tvsum_dict = hdf5_to_dict(tvsum_hdf)

    data_dict_list = {"summe": summe_dict, "tvsum": tvsum_dict}

    # 2 构造score_list
    score_list = {"summe": {}, "tvsum": {}}
    for dataset_name in data_dict_list.keys():
        for video_name in data_dict_list[dataset_name].keys():
            data_dict = data_dict_list[dataset_name]
            scores = np.zeros(int(data_dict[video_name]["n_steps"]))
            score_list[dataset_name][video_name] = scores

    # 3 将结果写入score_list
    result_list = {}
    pattern = r"Score:\s*(\d+\.\d+)"

    for file in result_file_list:
        with open(file, "r") as f:
            result = json.load(f)
        file_name = os.path.basename(file).split(".")[0]
        dataset_name = file_name.split("_")[0]
        check_out = []
        scores_list_tmp = copy.deepcopy(score_list[dataset_name])
        for sample in result:
            video_name = sample["id"].split(
                "_")[1] + "_" + sample["id"].split("_")[2]
            images = sample["images"]
            index = get_score_index_from_image_list(images, 15)
            index = np.array(index)
            llm_out = sample["llm_out"]
            scores = re.findall(pattern, llm_out)
            scores = [float(score) for score in scores]
            scores = np.array(scores)
            # 如果返回单个结果,转换为list
            if isinstance(scores, float):
                scores = [scores]
            # 检验scores个数,如果少于index个数,补齐0,多余的截断
            if len(scores) == len(index):
                check_out.append(1)
            else:
                if len(scores) < len(index):
                    # 补齐0
                    scores = np.concatenate(
                        (scores, np.zeros(len(index) - len(scores))))
                elif len(scores) > len(index):
                    scores = scores[: len(index)]
                check_out.append(0)
            # 写入score_list
            scores_list_tmp[video_name][index] = scores
            
        #  将 score 转换为 list 方便后面保存到 json
        for video_name in scores_list_tmp:
            scores_list_tmp[video_name] = scores_list_tmp[video_name].tolist()
        
        check_out = np.array(check_out)
        right = check_out.sum()
        length = len(check_out)
        print(f"rate:{right}/{length}")
        result_list[file_name] = scores_list_tmp

    # 4 保存score_list
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    scores_json = Path(scores_dir, "raw_llm_out_scores.json")
    with open(scores_json, "w") as f:
        json.dump(result_list, f, separators=(",", ": "), indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
    )
    parser.add_argument(
        "--scores_dir",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.result_dir, args.dataset_dir, args.scores_dir)

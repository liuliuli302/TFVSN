import argparse
from pathlib import Path
import json
from pprint import pprint
import torch
import numpy as np
import h5py
import copy
import os
import re


def knapSack(W, wt, val, n):
    K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1]
                              [w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    selected = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]

    return selected


def generate_single_summary(shot_bound, scores, n_frames, positions):
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])

    frame_scores = np.zeros(n_frames, dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = scores[i]

    # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
    shot_imp_scores = []
    shot_lengths = []
    for shot in shot_bound:
        shot_lengths.append(shot[1] - shot[0] + 1)
        shot_imp_scores.append(
            (frame_scores[shot[0]: shot[1] + 1].mean()).item())

    # Select the best shots using the knapsack implementation
    final_shot = shot_bound[-1]
    final_max_length = int((final_shot[1] + 1) * 0.15)

    selected = knapSack(
        final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths)
    )

    # Select all frames from each selected shot (by setting their value in the summary vector to 1)
    summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
    for shot in selected:
        summary[shot_bound[shot][0]: shot_bound[shot][1] + 1] = 1

    return summary


def evaluate_single_summary(predicted_summary, user_summary, eval_method):
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[: len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[: user_summary.shape[1]] = user_summary[user]
        overlapped = S & G

        # Compute precision, recall, f-score
        precision = sum(overlapped) / sum(S)
        recall = sum(overlapped) / sum(G)
        if precision + recall == 0:
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall *
                            100 / (precision + recall))

    if eval_method == "max":
        return max(f_scores)
    else:
        return sum(f_scores) / len(f_scores)


def get_fscore_from_predscore(
    pred_scores, shot_bound, n_frames, positions, user_summary, eval_method
):
    """Compute the F-score between the predicted summary and the user summary."""
    predicted_summary = generate_single_summary(
        shot_bound, pred_scores, n_frames, positions
    )
    return (
        evaluate_single_summary(predicted_summary, user_summary, eval_method),
        predicted_summary,
    )


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


def main(score_file, dataset_dir):
    summe_hdf = Path(dataset_dir, "SumMe", "summe.h5")
    tvsum_hdf = Path(dataset_dir, "TVSum", "tvsum.h5")
    summe_dict = hdf5_to_dict(summe_hdf)
    tvsum_dict = hdf5_to_dict(tvsum_hdf)

    data_dict_list = {"summe": summe_dict, "tvsum": tvsum_dict}

    with open(score_file) as f:
        result_list = json.load(f)

    # 4 得到fscore
    fscore_result = {}
    fscore_result_list = {}
    for key in result_list.keys():
        fscore_list = []
        result_scores = result_list[key]
        dataset_name = key.split("_")[0]
        if dataset_name == "summe":
            eval_method = "max"
        elif dataset_name == "tvsum":
            eval_method = "avg"
        dict_video = data_dict_list[dataset_name]
        for video_name in result_scores.keys():
            pred_scores = result_scores[video_name]
            n_frames = dict_video[video_name]["n_frames"]
            positions = dict_video[video_name]["picks"].astype(int)
            user_summary = dict_video[video_name]["user_summary"]
            shot_bound = dict_video[video_name]["change_points"].astype(int)
            f1score = get_fscore_from_predscore(
                pred_scores=pred_scores,
                shot_bound=shot_bound,
                n_frames=n_frames,
                positions=positions,
                user_summary=user_summary,
                eval_method=eval_method,
            )[0]
            fscore_list.append(f1score)
        fscore_list = np.array(fscore_list)
        avg_fscore = fscore_list.mean()
        fscore_result[key] = avg_fscore
        fscore_result_list[key] = fscore_list
    pprint(fscore_result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.score_file, args.dataset_dir)

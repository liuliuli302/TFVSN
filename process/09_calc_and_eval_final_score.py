import os
from pathlib import Path
import numpy as np
import json
import argparse
import sys
import h5py
import json
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
from PIL import Image
from tqdm import tqdm


class VideoSummarizationDataset(Dataset):
    def __init__(
        self,
        root_path="./data",
        dataset_name="SumMe",
    ):
        super().__init__()
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.dataset_dir = Path(root_path, dataset_name)
        self.frames_dir = Path(self.dataset_dir, "frames")

        self.data_list, self.video_name_dict = self._load_data()

        # Invert the values and keys in the self.video_name_dict
        self.video_name_dict_inv = {v: k for k,
                                    v in self.video_name_dict.items()}
        # pprint(self.video_name_dict_inv)

    def __len__(self):
        return len(self.video_name_dict)

    def __getitem__(self, idx):
        video_name_idx = f"video_{idx + 1}"
        video_name_real = self.video_name_dict_inv[video_name_idx]
        video_frames_dir = Path(self.frames_dir, video_name_real)

        video_info = self.data_list[video_name_idx]

        picks = video_info["picks"]
        keys = list(video_info.keys())
        # Convert picks to 6-digit integer.
        picks = [f"{pick:06d}" for pick in picks]
        # Gets all file names from picks.
        frame_file_paths = [
            str(Path(video_frames_dir, f"{pick}.jpg")) for pick in picks]

        video_info["frame_file_paths"] = frame_file_paths
        video_info["video_name"] = video_name_real

        # Debug info.
        # pprint(frame_file_paths)
        # pprint(keys)

        return video_info

    def _load_data(self):
        """
        Load data from `self.data_path`.
        """
        # 1 Load hdf file to dict.
        dataset_name_lower = self.dataset_name.lower()
        hdf_file_path = Path(self.dataset_dir, f"{dataset_name_lower}.h5")
        hdf_file = h5py.File(hdf_file_path, "r")

        hdf_dict = hdf5_to_dict(hdf_file)
        video_names = list(hdf_dict.keys())
        keys = list(hdf_dict["video_1"].keys())

        # 2 Load video_name dict.
        video_name_dict_file_path = Path(
            self.dataset_dir, "video_name_dict.json")
        with open(video_name_dict_file_path, "r") as f:
            video_name_dict = json.load(f)

        return hdf_dict, video_name_dict


def hdf5_to_dict(hdf5_file):
    def recursively_convert(h5_obj):
        if isinstance(h5_obj, h5py.Group):
            return {key: recursively_convert(h5_obj[key]) for key in h5_obj.keys()}
        elif isinstance(h5_obj, h5py.Dataset):
            return h5_obj[()]
        else:
            raise TypeError("Unsupported h5py object type")

    return recursively_convert(hdf5_file)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_score", type=str, required=True,
                        help="Path to llm_out_score.json")
    parser.add_argument("--sim_score_dir", type=str,
                        required=True, help="Path to similarity_scores_dir")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root dir")
    parser.add_argument("--output_dir", type=str,
                        required=True, help="Path to save results")
    return parser.parse_args()


def load_llm_out_scores(llm_score_path):
    with open(llm_score_path, 'r') as f:
        return json.load(f)


def load_similarity_scores(sim_dir):
    sim_scores = {}
    for dataset in ["SumMe", "TVSum"]:
        dataset_path = Path(sim_dir) / dataset
        sim_scores[dataset.lower()] = {}
        for file in dataset_path.glob("*.json"):
            video_name = file.stem
            with open(file, 'r') as f:
                sim_scores[dataset.lower()][video_name] = json.load(f)
    return sim_scores


def load_name_mapping(data_dir):
    name_mapping = {}
    for dataset in ["SumMe", "TVSum"]:
        with open(Path(data_dir) / dataset / "video_name_dict.json", 'r') as f:
            name_mapping[dataset.lower()] = json.load(f)
    return name_mapping


def main(llm_score_path, sim_score_dir, data_dir, output_dir):
    llm_scores = load_llm_out_scores(llm_score_path)
    sim_scores = load_similarity_scores(sim_score_dir)
    name_map = load_name_mapping(data_dir)

    # 反转 name_map，从 video_name -> video_i 变为 video_i -> video_name
    inv_name_map = {
        dataset: {v: k for k, v in mapping.items()}
        for dataset, mapping in name_map.items()
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    datasets = {"summe": "max", "tvsum": "avg"}
    llm_keys = ["jump", "turn"]
    sim_keys = ["max_p_max_m", "max_p_mean_m", "mean_p_max_m", "mean_p_mean_m"]
    alphas = [round(x, 1) for x in np.arange(0.1, 1.1, 0.1)]

    all_results = {}
    best_result = {}
    dataset_key_dict = {"summe": "SumMe", "tvsum": "TVSum"}
    for dataset_key, eval_method in datasets.items():
        dataset = VideoSummarizationDataset(data_dir, dataset_key_dict[dataset_key])
        best_score = -1
        best_combo = None

        all_results[dataset_key] = {}

        combos = [(llm_type, sim_type, alpha)
                  for llm_type in llm_keys for sim_type in sim_keys for alpha in alphas]

        for llm_type, sim_type, alpha in tqdm(combos, desc=f"Processing {dataset_key.upper()}"):
            result_key = f"{llm_type}+{sim_type}+alpha={alpha:.1f}"
            total_score = []
            all_results[dataset_key][result_key] = []

            llm_result_key = f"{dataset_key}_dataset_{llm_type}_result"
            video_items = list(llm_scores[llm_result_key].items())
            for video_i, llm_score in tqdm(video_items, desc=f"  Scoring {result_key}", leave=False):
                video_real_name = inv_name_map[dataset_key][video_i]
                sim_score_dict = sim_scores[dataset_key].get(
                    video_real_name, {})
                if sim_type not in sim_score_dict:
                    continue
                sim_score = sim_score_dict[sim_type]

                combined_score = alpha * \
                    np.array(llm_score) + (1 - alpha) * np.array(sim_score)
                combined_score = combined_score.tolist()
                id = int(video_i.split("_")[1]) - 1
                data = dataset[id]
                f1, _ = get_fscore_from_predscore(
                    combined_score,
                    data["change_points"],
                    data["n_frames"],
                    data["picks"],
                    data["user_summary"],
                    eval_method
                )
                total_score.append(f1)
                all_results[dataset_key][result_key].append(f1)

            avg_f1 = np.mean(total_score) if total_score else 0.0
            if avg_f1 > best_score:
                best_score = avg_f1
                best_combo = {
                    "llm_type": llm_type,
                    "sim_type": sim_type,
                    "alpha": alpha,
                    "f1": best_score
                }

        best_result[dataset_key] = best_combo

    with open(Path(output_dir) / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(Path(output_dir) / "best_results.json", "w") as f:
        json.dump(best_result, f, indent=2)

    print("✅ 所有组合结果已保存到 all_results.json")
    print("🏆 最佳组合已保存到 best_results.json")


if __name__ == "__main__":
    args = parse_args()
    main(args.llm_score, args.sim_score_dir, args.data_dir, args.output_dir)

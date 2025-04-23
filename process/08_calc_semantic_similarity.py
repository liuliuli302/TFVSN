import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path


def split_segments(data, num_segments, axis=0):
    length = data.shape[axis]
    seg_sizes = [(length + i) // num_segments for i in range(num_segments)]
    indices = np.cumsum(seg_sizes)[:-1]
    return torch.tensor_split(data, list(indices), axis)


def compute_similarity_scores(visual_feat, text_feat, segment_num):
    visual_segs = split_segments(visual_feat, segment_num, axis=0)
    text_segs = split_segments(text_feat, segment_num, axis=0)

    max_p_scores = []
    mean_p_scores = []
    max_m_scores = []
    mean_m_scores = []

    for vis_seg, txt_seg in zip(visual_segs, text_segs):
        scores = torch.einsum('md,npd->nmp', txt_seg, vis_seg)

        max_p = scores.max(dim=2).values
        mean_p = scores.mean(dim=2)

        max_p_max_m = max_p.max(dim=1).values.unsqueeze(1)
        max_p_mean_m = max_p.mean(dim=1).unsqueeze(1)
        mean_p_max_m = mean_p.max(dim=1).values.unsqueeze(1)
        mean_p_mean_m = mean_p.mean(dim=1).unsqueeze(1)

        max_p_scores.append(max_p_max_m)
        mean_p_scores.append(mean_p_max_m)
        max_m_scores.append(mean_p_max_m)
        mean_m_scores.append(mean_p_mean_m)

    score_dict = {
        'max_p_max_m': torch.cat(max_p_scores, dim=0),
        'mean_p_max_m': torch.cat(mean_p_scores, dim=0),
        'max_p_mean_m': torch.cat(max_m_scores, dim=0),
        'mean_p_mean_m': torch.cat(mean_m_scores, dim=0)
    }

    for k in score_dict:
        seq = score_dict[k]
        min_v = seq.min()
        max_v = seq.max()
        score_dict[k] = ((seq - min_v) / (max_v - min_v + 1e-8)
                         ).squeeze(1).tolist()

    return score_dict


def process_all(vfeat_dir: Path, tfeat_dir: Path, segment_num: int, out_put_dir: Path):
    out_put_dir.mkdir(parents=True, exist_ok=True)

    video_names = [
        f.stem for f in vfeat_dir.glob("*.npy")
        if (tfeat_dir / f"{f.stem}_text.npy").exists()
    ]

    for video in tqdm(video_names, desc='Processing videos'):
        vfeat_path = vfeat_dir / f"{video}.npy"
        tfeat_path = tfeat_dir / f"{video}_text.npy"

        vfeat = torch.from_numpy(np.load(vfeat_path))  # (n, p, d)
        tfeat = torch.from_numpy(np.load(tfeat_path))  # (m, d)

        score_result = compute_similarity_scores(vfeat, tfeat, segment_num)

        output_path = out_put_dir / f"{video}.json"
        with output_path.open('w') as f:
            json.dump(score_result, f, indent=2)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual_feature_dir', type=Path,
                        required=True, help='Path to visual features (.npy)')
    parser.add_argument('--text_feature_dir', type=Path,
                        required=True, help='Path to text features (.npy)')
    parser.add_argument('--segment_num', type=int,
                        required=True, help='Number of segments')
    parser.add_argument('--out_put_dir', type=Path, required=True,
                        help='Output directory for individual JSON files')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_all(args.visual_feature_dir, args.text_feature_dir,
                args.segment_num, args.out_put_dir)

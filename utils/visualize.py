#!/usr/bin/env python
# filepath: /root/TFVSN/utils/visualize.py
"""
视频摘要系统可视化工具
用于可视化pipeline处理结果
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2


def plot_score_distribution(scores_path, output_dir="visualization"):
    """绘制分数分布图
    
    Args:
        scores_path: 分数文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(scores_path, "r") as f:
        scores_data = json.load(f)
    
    for dataset_type, dataset in scores_data.items():
        # 创建数据集子目录
        dataset_dir = Path(output_dir) / dataset_type
        dataset_dir.mkdir(exist_ok=True)
        
        for video_name, scores in dataset.items():
            scores_array = np.array(scores)
            
            # 绘制分数分布图
            plt.figure(figsize=(12, 6))
            plt.plot(scores_array)
            plt.title(f"Score Distribution - {video_name}")
            plt.xlabel("Frame Index")
            plt.ylabel("Importance Score")
            plt.grid(True)
            plt.ylim(0, 1)
            
            # 保存图像
            plt.savefig(dataset_dir / f"{video_name}_scores.png")
            plt.close()
            
    print(f"分数分布图已保存到: {output_dir}")


def visualize_summary(video_path, scores_path, top_k=10, output_dir="visualization"):
    """可视化视频摘要
    
    Args:
        video_path: 视频文件路径
        scores_path: 分数文件路径
        top_k: 选择的顶部帧数量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # 读取分数
    with open(scores_path, "r") as f:
        scores_data = json.load(f)
        
    # 提取视频名称
    video_name = Path(video_path).stem
    
    # 查找对应的分数数组
    scores = None
    for dataset_type, dataset in scores_data.items():
        if video_name in dataset:
            scores = np.array(dataset[video_name])
            break
            
    if scores is None:
        print(f"Error: Cannot find scores for video {video_name}")
        return
        
    # 找到top_k个最重要的帧
    if len(scores) > top_k:
        top_indices = np.argsort(scores)[-top_k:]
    else:
        top_indices = np.arange(len(scores))
    
    # 提取并保存关键帧
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in top_indices:
            # 保存关键帧
            frame_path = Path(output_dir) / f"{video_name}_keyframe_{frame_count}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append(frame)
            
        frame_count += 1
        
    cap.release()
    
    # 创建拼接图像
    if frames:
        h, w, _ = frames[0].shape
        grid_size = int(np.ceil(np.sqrt(len(frames))))
        grid_h = grid_size * h
        grid_w = grid_size * w
        grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i, frame in enumerate(frames):
            row = i // grid_size
            col = i % grid_size
            grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = frame
            
        grid_path = Path(output_dir) / f"{video_name}_keyframes_grid.jpg"
        cv2.imwrite(str(grid_path), grid_img)
        
    print(f"关键帧可视化已保存到: {output_dir}")


def visualize_evaluation(eval_results_path, output_dir="visualization"):
    """可视化评估结果
    
    Args:
        eval_results_path: 评估结果文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(eval_results_path, "r") as f:
        eval_data = json.load(f)
        
    # 提取评估指标
    videos = list(eval_data.keys())
    f1_scores = [eval_data[v].get("f1_score", 0) for v in videos]
    precision = [eval_data[v].get("precision", 0) for v in videos]
    recall = [eval_data[v].get("recall", 0) for v in videos]
    
    # 计算平均值
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    
    # 绘制条形图
    x = np.arange(len(videos))
    width = 0.25
    
    plt.figure(figsize=(15, 8))
    plt.bar(x - width, f1_scores, width, label="F1 Score")
    plt.bar(x, precision, width, label="Precision")
    plt.bar(x + width, recall, width, label="Recall")
    
    plt.xlabel("Videos")
    plt.ylabel("Score")
    plt.title("Evaluation Results")
    plt.xticks(x, videos, rotation=45, ha="right")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    
    plt.savefig(Path(output_dir) / "evaluation_results.png")
    plt.close()
    
    # 保存平均值
    with open(Path(output_dir) / "average_metrics.json", "w") as f:
        json.dump({
            "average_f1_score": avg_f1,
            "average_precision": avg_precision,
            "average_recall": avg_recall
        }, f, indent=2)
        
    print(f"评估结果可视化已保存到: {output_dir}")
    print(f"平均F1分数: {avg_f1:.4f}")
    print(f"平均精确率: {avg_precision:.4f}")
    print(f"平均召回率: {avg_recall:.4f}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频摘要结果可视化")
    parser.add_argument("--scores", type=str, help="分数文件路径")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--eval", type=str, help="评估结果文件路径")
    parser.add_argument("--output", type=str, default="visualization", help="输出目录")
    parser.add_argument("--top_k", type=int, default=10, help="关键帧数量")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    if args.scores:
        plot_score_distribution(args.scores, args.output)
        
    if args.scores and args.video:
        visualize_summary(args.video, args.scores, args.top_k, args.output)
        
    if args.eval:
        visualize_evaluation(args.eval, args.output)
        
    if not (args.scores or args.eval):
        print("Error: 请至少提供--scores或--eval参数")


if __name__ == "__main__":
    main()

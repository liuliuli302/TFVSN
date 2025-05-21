#!/usr/bin/env python
# filepath: /root/TFVSN/main.py
"""
视频摘要系统主入口
用于配置和运行VideoSummarizationPipeline
"""

import os
import argparse
from pathlib import Path
import json

from pipeline import (
    VideoSummarizationPipeline, 
    VideoLoadStep,
    FrameExtractionStep, 
    SampleBuildStep,
    LLMAnalysisStep,
    DataCleaningStep,
    EvaluationStep
)
from video_processor import VideoProcessor
from handler import LLMHandler
from data_cleaner import DataCleaner
from samples_builder import SamplesBuilder
from config import VideoProcessorConfig
from evaluator import VideoSummarizationEvaluator


def create_pipeline(config):
    """创建并配置Pipeline
    
    Args:
        config: 配置字典
        
    Returns:
        配置好的VideoSummarizationPipeline实例
    """
    # 创建pipeline实例
    pipeline = VideoSummarizationPipeline()
    
    # 创建处理器实例
    video_processor = VideoProcessor(
        VideoProcessorConfig(config.get("video_path", ""))
    )
    llm_handler = LLMHandler(config.get("llm_model", "lmms-lab/LLaVA-Video-7B-Qwen2"))
    data_cleaner = DataCleaner(config.get("frame_interval", 15))
    
    # 添加处理步骤
    if config.get("process_video", True):
        pipeline.add_step(
            VideoLoadStep("video_load", video_processor)
        )
        pipeline.add_step(
            FrameExtractionStep("frame_extraction", video_processor, {
                "frames_dir": config.get("frames_dir", "frames"),
                "sample_rate": config.get("sample_rate", 15)
            })
        )
    
    if config.get("build_samples", True):
        pipeline.add_step(
            SampleBuildStep("sample_build", SamplesBuilder, {
                "clip_length": config.get("clip_length", 5),
                "mode": config.get("mode", "both")
            })
        )
    
    if config.get("run_llm_analysis", True):
        pipeline.add_step(
            LLMAnalysisStep("llm_analysis", llm_handler, {
                "result_dir": config.get("result_dir", "results"),
                # 不再使用batch_size参数
            })
        )
    
    if config.get("clean_data", True):
        pipeline.add_step(
            DataCleaningStep("data_cleaning", data_cleaner, {
                "dataset_dir": config.get("dataset_dir", "dataset"),
                "scores_dir": config.get("scores_dir", "scores")
            })
        )
    
    if config.get("evaluate", True):
        pipeline.add_step(
            EvaluationStep("evaluation", None, {
                "eval_dir": Path(config.get("output_dir", "output")) / "evaluation"
            })
        )
    
    # 设置上下文数据
    if config.get("ground_truth_path"):
        with open(config["ground_truth_path"], "r") as f:
            ground_truth = json.load(f)
        pipeline.set_context("ground_truth", ground_truth)
    
    pipeline.set_context("data_dir", config.get("data_dir", "."))
    
    return pipeline


def load_config(config_path):
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not config_path:
        return {}
        
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频摘要系统")
    parser.add_argument("--config", type=str, default="config_pipeline.json", help="配置文件路径")
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 确保输出目录存在
    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保输出目录存在
    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建pipeline
    pipeline = create_pipeline(config)
    
    # 确定输入数据
    input_data = config.get("video_path", "")
    if not input_data and config.get("data_dir"):
        # 如果没有指定视频路径但指定了数据目录，使用数据目录
        input_data = config["data_dir"]
    
    # 运行pipeline
    if not input_data:
        print("错误：未指定输入数据。请提供--video_path或--data_dir参数。")
        return
    
    try:
        results = pipeline.run(input_data, save_path=output_dir)
        print("处理完成！")
        print(f"结果保存在: {output_dir}")
        
        # 打印评估结果（如果有）
        if "evaluation" in results and results["evaluation"]:
            print("\n评估结果:")
            print(json.dumps(results["evaluation"], indent=2))
            
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# filepath: /root/TFVSN/examples/pipeline_example.py
"""
视频摘要系统使用示例
展示如何直接在代码中配置和使用VideoSummarizationPipeline
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from pipeline import (
    VideoSummarizationPipeline, 
    VideoLoadStep,
    FrameExtractionStep, 
    SampleBuildStep,
    LLMAnalysisStep,
    DataCleaningStep
)
from video_processor import VideoProcessor
from handler import LLMHandler
from data_cleaner import DataCleaner
from samples_builder import SamplesBuilder
from config import VideoProcessorConfig


def example_basic_pipeline():
    """基础视频摘要pipeline示例"""
    # 创建pipeline实例
    pipeline = VideoSummarizationPipeline()
    
    # 创建处理器实例
    video_config = VideoProcessorConfig("/path/to/video")
    video_config.frame_rate = 30
    video_config.resolution = (1280, 720)
    
    video_processor = VideoProcessor(video_config)
    llm_handler = LLMHandler("lmms-lab/LLaVA-Video-7B-Qwen2")
    data_cleaner = DataCleaner(frame_interval=15)
    
    # 配置pipeline步骤
    pipeline.add_step(
        VideoLoadStep("video_load", video_processor)
    )
    pipeline.add_step(
        FrameExtractionStep("frame_extraction", video_processor, {
            "frames_dir": "output/frames",
            "sample_rate": 15
        })
    )
    pipeline.add_step(
        SampleBuildStep("sample_build", SamplesBuilder, {
            "clip_length": 5,
            "mode": "both"
        })
    )
    pipeline.add_step(
        LLMAnalysisStep("llm_analysis", llm_handler, {
            "result_dir": "output/results"
        })
    )
    pipeline.add_step(
        DataCleaningStep("data_cleaning", data_cleaner, {
            "dataset_dir": "dataset",
            "scores_dir": "output/scores"
        })
    )
    
    # 设置上下文
    pipeline.set_context("data_dir", "dataset")
    
    # 运行pipeline
    results = pipeline.run("/path/to/video", save_path="output")
    return results


def example_custom_pipeline():
    """自定义视频摘要pipeline示例"""
    # 创建pipeline实例
    pipeline = VideoSummarizationPipeline()
    
    # 创建处理器实例
    video_processor = VideoProcessor(VideoProcessorConfig(""))
    llm_handler = LLMHandler()
    
    # 仅添加需要的步骤
    pipeline.add_step(
        LLMAnalysisStep("llm_analysis", llm_handler, {
            "result_dir": "output/results"
        })
    )
    
    # 设置上下文
    pipeline.set_context("data_dir", "dataset")
    
    # 运行pipeline(直接处理已有的样本数据)
    existing_samples_path = "path/to/existing_samples.json"
    results = pipeline.run(existing_samples_path, save_path="output")
    return results


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    
    print("运行基础pipeline示例...")
    # example_basic_pipeline()  # 取消注释以运行示例
    
    print("运行自定义pipeline示例...")
    # example_custom_pipeline()  # 取消注释以运行示例
    
    print("完成!")

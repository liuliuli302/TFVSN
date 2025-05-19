from typing import Dict, List, Any, Union, Optional, Callable
from pathlib import Path
import os
import json
from datetime import datetime

from video_processor import VideoProcessor
from handler import LLMHandler
from data_cleaner import DataCleaner
from evaluator import VideoSummarizationEvaluator
from samples_builder import SamplesBuilder
from config import VideoProcessorConfig


class PipelineStep:
    """Pipeline步骤的基础类"""
    def __init__(self, name: str, processor: Any, config: dict = None):
        """初始化pipeline步骤
        
        Args:
            name: 步骤名称
            processor: 处理器实例
            config: 步骤配置
        """
        self.name = name
        self.processor = processor
        self.config = config or {}
        
    def process(self, data: Any, context: dict) -> Any:
        """处理数据的抽象方法
        
        Args:
            data: 输入数据
            context: pipeline上下文
            
        Returns:
            处理后的数据
        """
        raise NotImplementedError


class VideoLoadStep(PipelineStep):
    """视频加载步骤"""
    def process(self, data: str, context: dict) -> Dict[str, Any]:
        video_path = data
        videos = self.processor.load_multi_videos(video_path)
        context["video_data"] = videos
        return videos


class FrameExtractionStep(PipelineStep):
    """帧提取步骤"""
    def process(self, data: Dict[str, Any], context: dict) -> List[Any]:
        frames_dir = self.config.get("frames_dir", "frames")
        sample_rate = self.config.get("sample_rate", 1)
        os.makedirs(frames_dir, exist_ok=True)
        
        results = []
        for video_name, video_data in data.items():
            video_frames_dir = Path(frames_dir) / video_name
            result = self.processor.extract_frames(
                str(video_frames_dir), 
                str(video_frames_dir), 
                sample_rate
            )
            results.extend(result)
            
        context["frame_info"] = results
        return results


class SampleBuildStep(PipelineStep):
    """样本构建步骤"""
    def process(self, data: List[Any], context: dict) -> List[Dict[str, Any]]:
        clip_length = self.config.get("clip_length", 5)
        mode = self.config.get("mode", "both")
        
        samples = self.processor(
            data_dir=context.get("data_dir", "."),
            clip_length=clip_length,
            mode=mode
        )
        
        context["samples"] = samples
        return samples


class LLMAnalysisStep(PipelineStep):
    """LLM分析步骤"""
    def process(self, data: List[Dict[str, Any]], context: dict) -> str:
        result_dir = self.config.get("result_dir", "results")
        os.makedirs(result_dir, exist_ok=True)
        
        result_path = self.processor.process_dataset(
            data, 
            result_dir
        )
        
        context["llm_results"] = result_path
        return result_path


class DataCleaningStep(PipelineStep):
    """数据清理步骤"""
    def process(self, data: str, context: dict) -> str:
        dataset_dir = self.config.get("dataset_dir", "dataset")
        scores_dir = self.config.get("scores_dir", "scores")
        
        scores_path = self.processor.clean_llm_outputs(
            data,
            dataset_dir,
            scores_dir
        )
        
        context["cleaned_data"] = scores_path
        return scores_path


class EvaluationStep(PipelineStep):
    """评估步骤"""
    def process(self, data: str, context: dict) -> Dict[str, float]:
        with open(data, 'r') as f:
            predictions = json.load(f)
            
        eval_results = {}
        for video_name, pred in predictions.items():
            ground_truth = context.get("ground_truth", {}).get(video_name)
            if ground_truth is not None:
                evaluator = VideoSummarizationEvaluator(ground_truth, pred)
                eval_results[video_name] = evaluator.evaluate_comprehensive()
                
        context["evaluation"] = eval_results
        return eval_results


class VideoSummarizationPipeline:
    """视频摘要处理流水线"""
    
    def __init__(self):
        """初始化pipeline"""
        self.steps = []
        self.context = {}
        self.results_history = []
        
    def add_step(self, step: PipelineStep) -> 'VideoSummarizationPipeline':
        """添加处理步骤
        
        Args:
            step: pipeline步骤实例
            
        Returns:
            self，支持链式调用
        """
        self.steps.append(step)
        return self
    
    def remove_step(self, step_name: str) -> None:
        """移除指定名称的处理步骤
        
        Args:
            step_name: 要移除的步骤名称
        """
        self.steps = [step for step in self.steps if step.name != step_name]
        
    def get_step(self, step_name: str) -> Optional[PipelineStep]:
        """获取指定名称的处理步骤
        
        Args:
            step_name: 步骤名称
            
        Returns:
            找到的步骤实例，如果不存在则返回None
        """
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
        
    def run(self, input_data: Any, save_path: Optional[str] = None) -> Dict[str, Any]:
        """运行pipeline
        
        Args:
            input_data: 输入数据
            save_path: 结果保存路径，可选
            
        Returns:
            Dict[str, Any]: 处理结果和中间结果
        """
        current_data = input_data
        step_results = {}
        
        try:
            for step in self.steps:
                print(f"Running step: {step.name}")
                current_data = step.process(current_data, self.context)
                step_results[step.name] = current_data
                
            self.results_history.append({
                'timestamp': datetime.now().isoformat(),
                'results': step_results
            })
            
            if save_path:
                self._save_results(save_path, step_results)
                
            return step_results
            
        except Exception as e:
            print(f"Error in step {step.name}: {str(e)}")
            raise
        
    def _save_results(self, save_path: str, results: Dict[str, Any]) -> None:
        """保存处理结果
        
        Args:
            save_path: 保存路径
            results: 处理结果
        """
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = Path(save_path) / f"pipeline_results_{timestamp}.json"
        
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
            
    def get_context(self) -> dict:
        """获取pipeline上下文
        
        Returns:
            dict: 当前上下文
        """
        return self.context
    
    def set_context(self, key: str, value: Any) -> None:
        """设置上下文值
        
        Args:
            key: 键
            value: 值
        """
        self.context[key] = value
        
    def clear_context(self) -> None:
        """清空上下文"""
        self.context.clear()
from typing import Dict, List, Any, Union, Optional, Callable
from pathlib import Path
import os
import json
from datetime import datetime
from tqdm import tqdm

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
        
        # 创建SamplesBuilder实例
        samples_builder = self.processor(
            data_dir=context.get("data_dir", "."),
            clip_length=clip_length,
            mode=mode
        )
        
        # 将Dataset转换为普通列表，使其可JSON序列化
        samples = []
        for i in range(len(samples_builder)):
            sample = samples_builder[i]
            # 确保每个样本有唯一ID
            if "id" not in sample:
                sample_id = f"{sample.get('dataset_name', 'unknown')}_{sample.get('video_name', 'unknown')}_{sample.get('mode', 'unknown')}_{i}"
                sample["id"] = sample_id
            samples.append(sample)
            
        context["samples"] = samples
        return samples


class LLMAnalysisStep(PipelineStep):
    """LLM分析步骤"""
    def process(self, data: List[Dict[str, Any]], context: dict) -> str:
        result_dir = self.config.get("result_dir", "results")
        # 创建结果目录结构，包括中间结果目录
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(Path(result_dir) / "intermediate", exist_ok=True)
        
        # 处理数据集，支持恢复处理并保存中间结果
        print(f"开始LLM分析，结果将保存在 {result_dir} 目录")
        result_path = self.processor.process_dataset(
            data,
            result_dir
        )
        
        # 将结果路径添加到上下文中
        context["llm_results"] = result_path
        return result_path


class DataCleaningStep(PipelineStep):
    """数据清理步骤"""
    def process(self, data: str, context: dict) -> str:
        dataset_dir = self.config.get("dataset_dir", "dataset")
        scores_dir = self.config.get("scores_dir", "scores")
        
        # 创建存储分数的目录
        os.makedirs(scores_dir, exist_ok=True)
        intermediate_dir = Path(scores_dir) / "intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        
        print(f"开始清理和处理LLM输出数据，中间结果将保存在 {intermediate_dir} 目录")
        
        # 运行数据清理并处理，支持恢复处理
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
        # 创建评估结果目录
        eval_dir = self.config.get("eval_dir", "evaluation_results")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 加载预测数据
        with open(data, 'r') as f:
            predictions = json.load(f)
        
        # 检查是否有现有的评估结果
        result_file = Path(eval_dir) / "evaluation_results.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    eval_results = json.load(f)
                    print(f"已加载现有评估结果: {result_file}")
                    context["evaluation"] = eval_results
                    return eval_results
            except (json.JSONDecodeError, KeyError) as e:
                print(f"无法加载现有评估结果: {e}")
        
        # 逐个评估视频
        eval_results = {}
        eval_count = 0
        
        for video_name, pred in tqdm(predictions.items(), desc="评估视频", unit="video"):
            ground_truth = context.get("ground_truth", {}).get(video_name)
            if ground_truth is not None:
                try:
                    evaluator = VideoSummarizationEvaluator(ground_truth, pred)
                    eval_results[video_name] = evaluator.evaluate_comprehensive()
                    eval_count += 1
                    
                    # 每完成5个评估保存一次中间结果
                    if eval_count % 5 == 0:
                        with open(result_file, 'w') as f:
                            json.dump(eval_results, f, indent=2)
                except Exception as e:
                    print(f"评估视频 {video_name} 时出错: {e}")
        
        # 保存最终评估结果
        with open(result_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
            
        print(f"评估完成，已保存结果到: {result_file}")
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
        """运行pipeline，支持中间结果的保存和故障恢复
        
        Args:
            input_data: 输入数据
            save_path: 结果保存路径，可选
            
        Returns:
            Dict[str, Any]: 处理结果和中间结果
        """
        current_data = input_data
        step_results = {}
        
        # 创建用于存储中间结果的目录
        if save_path:
            checkpoint_dir = Path(save_path) / "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 尝试读取检查点
            checkpoint_file = checkpoint_dir / "pipeline_checkpoint.json"
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, "r") as f:
                        checkpoint_data = json.load(f)
                        last_completed_step = checkpoint_data.get("last_completed_step")
                        if last_completed_step:
                            print(f"发现检查点，上次处理到步骤: {last_completed_step}")
                            # 在实际生产环境中，可以添加代码来恢复上次的处理状态
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"无法读取检查点文件: {e}")
        
        # 开始处理流水线步骤
        for i, step in enumerate(tqdm(self.steps, desc="Pipeline Progress", unit="step")):
            try:
                print(f"运行步骤: {step.name}")
                current_data = step.process(current_data, self.context)
                step_results[step.name] = current_data
                
                # 保存当前步骤结果作为检查点
                if save_path:
                    # 保存检查点信息
                    checkpoint_info = {
                        "last_completed_step": step.name,
                        "completed_step_index": i,
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(checkpoint_dir / "pipeline_checkpoint.json", "w") as f:
                        json.dump(checkpoint_info, f, indent=2)
                    
                    # 保存中间结果
                    step_result_file = checkpoint_dir / f"step_{i}_{step.name}_result.json"
                    self._save_step_result(step_result_file, current_data)
                    
                    print(f"已保存步骤 {step.name} 的中间结果和检查点")
                        
            except Exception as e:
                print(f"步骤 {step.name} 出错: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # 保存已完成步骤的结果
                if save_path:
                    self._save_results(save_path, step_results)
                    print(f"错误发生前的结果已保存到: {save_path}")
                
                # 自动继续执行下一步，不需要手动确认
                print(f"将继续执行下一步骤...")
                continue
        
        # 记录历史
        self.results_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': step_results
        })
        
        # 保存完整结果
        if save_path:
            self._save_results(save_path, step_results)
                
        return step_results
        
    def _save_results(self, save_path: str, results: Dict[str, Any]) -> None:
        """保存处理结果
        
        Args:
            save_path: 保存路径
            results: 处理结果
        """
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = Path(save_path) / f"pipeline_results_{timestamp}.json"
        
        # 处理非JSON序列化对象
        serializable_results = {}
        for key, value in results.items():
            try:
                # 尝试JSON序列化，检查是否可序列化
                json.dumps(value)
                serializable_results[key] = value
            except (TypeError, OverflowError):
                # 如果不可序列化，转换为字符串表示
                if isinstance(value, (list, tuple)):
                    # 处理列表/元组：尝试逐项序列化
                    try:
                        serializable_items = []
                        for item in value:
                            try:
                                # 尝试直接序列化项目
                                json.dumps(item)
                                serializable_items.append(item)
                            except (TypeError, OverflowError):
                                # 如果项目不可序列化，使用字符串表示
                                serializable_items.append(str(item))
                        serializable_results[key] = serializable_items
                    except Exception:
                        # 如果上述处理失败，使用字符串表示
                        serializable_results[key] = f"<非JSON可序列化对象: {type(value).__name__}>"
                else:
                    # 非列表对象，使用字符串表示
                    serializable_results[key] = f"<非JSON可序列化对象: {type(value).__name__}>"
        
        # 保存处理后的结果
        with open(result_file, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
            
    def _save_step_result(self, file_path: Path, data: Any) -> None:
        """保存步骤中间结果，支持不同类型的数据
        
        Args:
            file_path: 保存路径
            data: 处理结果数据
        """
        try:
            # 处理字符串类型（通常是文件路径）
            if isinstance(data, (str, bytes, os.PathLike)):
                # 如果数据是路径，保存路径信息
                with open(file_path, "w") as f:
                    json.dump({"file_path": str(data)}, f, indent=2)
                return
                
            # 处理字典、列表等可JSON序列化的对象
            if isinstance(data, (dict, list, int, float, bool)) or data is None:
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                return
                
            # 处理其他类型的对象（保存为字符串表示）
            with open(file_path, "w") as f:
                json.dump({"data_repr": str(data)}, f, indent=2)
                
        except Exception as e:
            print(f"无法保存中间结果到 {file_path}: {e}")
            # 记录错误但不中断流程
            
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
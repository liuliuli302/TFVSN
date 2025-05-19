# 视频摘要系统 (TFVSN)

本项目实现了一个模块化、可扩展的视频摘要系统，使用大型语言模型(LLM)分析视频内容并生成摘要。

## 关键组件

+ **VideoProcessor**：用来处理视频、提取视频帧，或者将一个或者多个视频读取到Tensor中

+ **LLMHandler**：用于封装大型语言模型，实现图像分析和摘要生成功能

+ **SamplesBuilder**：创建用于送入LLM的样本数据集

+ **DataCleaner**：清理和处理LLM的输出结果

+ **Evaluator**：评估视频摘要的质量，包含多种评价指标

+ **VideoSummarizationPipeline**：整合所有组件的流水线，实现端到端的视频摘要生成

## 项目特点

1. **模块化设计**：各组件职责明确，可独立开发和测试
2. **高度可扩展**：易于添加新的处理步骤和功能
3. **灵活配置**：支持通过配置文件和命令行参数调整系统行为
4. **结果可视化**：提供工具可视化处理结果和评估指标

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用脚本运行

最简单的方式是使用提供的 shell 脚本:

```bash
# 使用默认配置运行 pipeline
./run_pipeline.sh

# 指定配置文件和数据目录
./run_pipeline.sh --config config_pipeline.json --data /path/to/data --output /path/to/output

# 处理特定视频
./run_pipeline.sh --video /path/to/video.mp4

# 跳过特定步骤
./run_pipeline.sh --skip-video --skip-llm
```

### 使用 Python 代码运行

```python
# main.py 提供命令行接口
python main.py --config config_pipeline.json --data_dir dataset --output_dir output

# 也可以在自己的代码中使用 pipeline
from pipeline import (
    VideoSummarizationPipeline, 
    VideoLoadStep,
    FrameExtractionStep,
    LLMAnalysisStep
)
from video_processor import VideoProcessor
from handler import LLMHandler
from config import VideoProcessorConfig

# 创建 pipeline
pipeline = VideoSummarizationPipeline()

# 创建处理器
video_processor = VideoProcessor(VideoProcessorConfig("/path/to/video"))
llm_handler = LLMHandler()

# 添加处理步骤
pipeline.add_step(VideoLoadStep("video_load", video_processor))
pipeline.add_step(LLMAnalysisStep("llm_analysis", llm_handler))

# 运行 pipeline
results = pipeline.run("input_path", save_path="output")
```

## 可视化结果

使用提供的可视化工具查看结果:

```bash
# 可视化分数分布
python utils/visualize.py --scores path/to/scores.json

# 可视化关键帧
python utils/visualize.py --scores path/to/scores.json --video path/to/video.mp4

# 可视化评估结果
python utils/visualize.py --eval path/to/eval_results.json
```

## 扩展 Pipeline

添加新的处理步骤:

1. 创建继承自 `PipelineStep` 的新类
2. 实现 `process` 方法
3. 将步骤添加到 pipeline

```python
from pipeline import PipelineStep

# 定义新的处理步骤
class MyCustomStep(PipelineStep):
    def process(self, data, context):
        # 处理逻辑
        processed_data = self.processor.do_something(data)
        context["my_results"] = processed_data
        return processed_data

# 添加到 pipeline
pipeline.add_step(MyCustomStep("my_step", my_processor, {"option": "value"}))
```




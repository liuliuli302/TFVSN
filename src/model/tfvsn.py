from pathlib import Path
import torch
import torch.nn as nn
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
    AutoModel,
)


class TrainingFreeVideoSummarizationNetwork(nn.Module):
    def __init__(self, model_name_or_path):
        super(TrainingFreeVideoSummarizationNetwork, self).__init__()
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False
        )

        self.tokenizer = self.model.update_special_tokens(tokenizer)

        self.vision_encoder = self.model.vlm.vision_encoder
        self.vision_tokenizer = self.model.vlm.vision_tokenizer

        self.lang_model = self.model.vlm.lang_model

    def _get_vision_tokens(self, image_path):
        """
        Using `self.vision_encoder` and `self.vision_tokenizer` to get vision tokens from image.
        可能比较复杂, 请参考`class: XGenMMPerceiver`里的generate方法实现
        """
        # 1 预处理Image
        # 2 使用`self.vision_encoder`提取特征
        # 3 使用`self.vision_tokenizer`生成tokens
        pass

    def _get_text_tokens(self, text):
        """
        Using `self.lang_model` to get text tokens.
        根据文本生成文本tokens
        """
        # 1 使用`self.lang_model`生成tokens
        pass

    def _get_scores_from_llm_output(self, llm_output):
        """
        Get scores from llm output.
        从LLM的输出中提取分数
        """
        # 1 从LLM的输出中提取分数
        pass

    def forward(self, video_data, prompt):
        """
        Forward pass of the model.
        输入为一个视频的所有图像帧和一个文本提示，输出为(LLM生成的文本序列, 提取得到的分数序列)
        """
        pass


def hook_modify_vision_input_of_llm(model, inputs):
    """
    A Hook Function.
    Modify vision input of llm. Add `position embeddings` to vision tokens.
    可以考虑用lambda函数实现在类中
    """
    pass


if __name__ == "__main__":
    model_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    tf_model = TrainingFreeVideoSummarizationNetwork(model_path)
    print("HELLO WORLD!")

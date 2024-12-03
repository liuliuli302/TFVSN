import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import PIL
import torch
import torch.nn as nn
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
)
import IPython.display as display
from IPython.display import Image

sys.path.append("/home/insight/workspace/TFVSN/src/")
from dataloader import VideoSummarizationDataset


class TrainingFreeVideoSummarizationNetwork(nn.Module):
    def __init__(self, model_name_or_path):
        super(TrainingFreeVideoSummarizationNetwork, self).__init__()
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False
        )
        self.tokenizer = self.model.update_special_tokens(tokenizer)

        self.vision_encoder = self.model.vlm.vision_encoder
        self.vision_tokenizer = self.model.vlm.vision_tokenizer

        self.lang_model = self.model.vlm.lang_model

        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

    def _get_vision_model_input(
        self,
        image_paths: List[Path],
    ):
        """
        Organizes the data in the dataloader into a form acceptable to the vision encoder.
        """
        image_list = []
        image_sizes = []
        for fn in image_paths:
            img = PIL.Image.open(fn)
            display.display(Image(filename=fn, width=300))
            image_list.append(
                self.image_processor([img], image_aspect_ratio="anyres")["pixel_values"]
                .to('cpu')
                .bfloat16()
            )
            image_sizes.append(img.size)
        inputs = {"pixel_values": [image_list], "image_sizes": [image_sizes]}
        return inputs

    def _get_vision_tokens(
        self,
        vision_x: torch.Tensor,
        device: str,
        image_size: Optional[Tuple] = None,
        **kwargs,
    ):
        """
        Using `self.vision_encoder` and `self.vision_tokenizer` to get vision tokens from image.
        可能比较复杂, 请参考`class: XGenMMPerceiver`里的generate方法实现
        """
        num_beams = kwargs.pop("num_beams", 1)
        # convert pixels to vision tokens
        vision_attention_mask = None
        if self.model.vlm.image_aspect_ratio == "anyres":
            input_dict = dict(image=vision_x, image_size=image_size)
            vision_features, vision_attn_masks = self.model.vlm._encode_vision_x_anyres(
                input_dict, device
            )
        else:
            vision_features = self.model.vlm._encode_vision_x(vision_x=vision_x)
            vision_attn_masks = None
        # If doing patch sampling, then flatten patches of shape [b, Np_i, v, d] -> [b*Np, v, d]
        # Same for attention masks: [b, Np, v] -> [b*Np, v]
        if self.anyres_patch_sampling:
            split_sizes = [feature.shape[0] for feature in vision_features]
            # Nested splits for multi-image samples.
            if isinstance(vision_x[0], list):
                nt_images = [len(images) for images in vision_x]
                split_split_sizes = []
                img_id = 0
                for nt in nt_images:
                    split_split_sizes.append(split_sizes[img_id : img_id + nt])
                    img_id += nt
            else:
                nt_images = [1] * len(vision_x)
                split_split_sizes = split_sizes
            vision_features = torch.cat(vision_features, dim=0)
            vision_features = vision_features[:, None, None, :, :]  # Expand dimensions.
            vision_attn_masks = torch.cat(vision_attn_masks, dim=0)
        vision_tokens = self.model.vlm.vision_tokenizer(vision_features, vision_attn_masks)

        # Post-processing: Split the batches into groups of patches and concatenate them together.
        if self.vlm.anyres_patch_sampling:
            assert isinstance(vision_x, list)
            if isinstance(vision_x[0], list):
                vision_token_groups = torch.split(
                    vision_tokens,
                    list(sum(nt_img) for nt_img in split_split_sizes),
                    dim=0,
                )
                vision_tokens = []

                for sample_id, patch_vis_tokens in enumerate(vision_token_groups):
                    patch_vis_token_groups = torch.split(
                        patch_vis_tokens, split_split_sizes[sample_id], dim=0
                    )  # [Np*nt, 1, v, d] -> [[Np_t, 1, v, d], ...]
                    flatten_vision_tokens = []
                    for image_vis_token in patch_vis_token_groups:
                        image_vis_token = image_vis_token.flatten(
                            0, 2
                        )  # [Np, 1, v, d] -> [Np*v, d]
                        flatten_vision_tokens.append(image_vis_token)
                    vision_tokens_i = flatten_vision_tokens
                    vision_tokens.append(vision_tokens_i)
            else:
                vision_token_groups = torch.split(vision_tokens, split_sizes, dim=0)
                vision_tokens = []
                for patch_vis_tokens in vision_token_groups:
                    patch_vis_tokens = patch_vis_tokens.flatten(
                        0, 2
                    )  # [Np, 1, v, d] -> [Np*v, d]
                    vision_tokens.append(
                        patch_vis_tokens.unsqueeze(0)
                    )  # Add the nt dimension.
        return vision_tokens

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
    dataset = VideoSummarizationDataset()
    data_sample = dataset[0]
    frame_paths = data_sample["frame_file_paths"]

    temp_input = frame_paths[0:2]
    
    inputs = tf_model._get_vision_model_input(temp_input)
    out = tf_model._get_vision_tokens(
        vision_x=inputs["pixel_values"],
        device="cpu",
        image_size=inputs['image_sizes']
    )
 
    print(out.shape)
    
    print("HELLO WORLD!")

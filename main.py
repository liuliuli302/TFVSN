import PIL
import json
import textwrap
from IPython.display import Image
import IPython.display as display
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataloader import VideoSummarizationDataset
from src.model import TrainingFreeVideoSummarizationNetwork


def apply_prompt_template(prompt):
    s = (
        "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    )
    return s


def main():
    context_prompt = "You are a professional media worker and director, and you need to edit a film to retain the main content. Now you are given a frame, and you need to decide whether to keep the frame and score the importance of the frame from 0 to 1, 0 means keep it and 1 means remove it."
    format_prompt = "Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'."
    summary_prompt = "Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."
    model_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    tf_model = TrainingFreeVideoSummarizationNetwork(model_path)
    vs_dataset = VideoSummarizationDataset(root_path="./data", dataset_name="SumMe")
    dataloader = DataLoader(vs_dataset, batch_size=1, shuffle=False)
    for sample in dataloader:
        image_list = []
        image_sizes = []
        image_paths = sample["frame_file_paths"]
        for fn in image_paths:
            # fn = os.path.join('/export/home/manlis/xgen-mm-phi3-mini-instruct-r-v1.5', fn)
            img = PIL.Image.open(fn)
            display.display(Image(filename=fn, width=300))
            image_list.append(
                tf_model.image_processor([img], image_aspect_ratio="anyres")[
                    "pixel_values"
                ]
                .cuda()
                .bfloat16()
            )
            image_sizes.append(img.size)
            inputs = {"pixel_values": [image_list]}

            for query in sample["question"]:
                prompt = apply_prompt_template(query)
                language_inputs = tf_model.tokenizer([prompt], return_tensors="pt")
                inputs.update(language_inputs)
                # To cuda
                for name, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        inputs[name] = value.cuda()
                generated_text = tf_model.model.generate(
                    **inputs,
                    image_size=[image_sizes],
                    pad_token_id=tf_model.tokenizer.pad_token_id,
                    eos_token_id=tf_model.tokenizer.eos_token_id,
                    temperature=0.05,
                    do_sample=False,
                    max_new_tokens=1024,
                    top_p=None,
                    num_beams=1,
                )
                prediction = tf_model.tokenizer.decode(
                    generated_text[0], skip_special_tokens=True
                ).split("<|end|>")[0]
                print("User: ", query)
                print("Assistant: ", textwrap.fill(prediction, width=100))
            print("-" * 120)


def extract_and_save_features():
    dataset_name="TVSum"
    
    
    
    model_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    tf_model = TrainingFreeVideoSummarizationNetwork(model_path).to("cuda").eval()
    dataset = VideoSummarizationDataset(
        root_path="./data",
        dataset_name=dataset_name,
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Extracting and saving vision features...")
    for sample in tqdm(data_loader, desc="Extract and save vison features"):
        tf_model.extract_and_save_vison_feature(
            dataset_dir=f"/home/insight/workspace/TFVSN/data/{dataset_name}",
            video_name=sample["video_name"],
            frame_paths=sample["frame_file_paths"]
        )


if __name__ == "__main__":
    extract_and_save_features()

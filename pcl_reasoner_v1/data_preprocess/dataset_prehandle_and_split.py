import os
import json
import pickle
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from transformers import AutoTokenizer

system = "<|im_start|>system\nYou are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<|im_end|>\n"
user_prompt = "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"
user_prompt_math = "<|im_start|>user\n{{content}}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"


def convert_json(json_file):
    dict_dataset = []
    token_length_collect = []
    print(f"Handling original dataset file {json_file} ...")
    cache_file = json_file.split(".")[0] + "_cache.dat"
    if not os.path.exists(cache_file):
        with open(json_file) as f:
            data = [json.loads(line) for line in f]
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"save cache file to {cache_file}")
    else:
        print(f"Found cache file {cache_file}, loading from cache file...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print("Done loading from cache file.")
    
    print("Convert json to dataset...")
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    print("Done converting json to dataset.")
    
    for data in tqdm(dataset, total=len(dataset), miniters=100):
        dict_data = {}
        dict_data['instruction'] = data['conversations'][0]['value']
        dict_data['input'] = ""
        output = data['conversations'][1]['value']
        output = output.replace("<think>", "<think>\n")
        output = output.replace("</answer>", "\n</answer>")
        dict_data['output'] = output
        token_length = len(tokenizer.tokenize(dict_data['output']))
        dict_dataset.append(dict_data)
        token_length_collect.append(token_length)
    
    print("Statistic the token length distribution ...")
    plt.hist(token_length_collect, bins=100)
    token_length_file = json_file.split(".")[0] + "_corr_token_length.txt"
    with open(token_length_file, "w") as f:
        f.write(str(token_length_collect))
        
    dist_image = json_file.split(".")[0] + "_corr_token_length_dist.png"
    plt.savefig(dist_image)
    print("Token distribution saved to:", dist_image, "for", json_file, "file")
    
    new_dataset = Dataset.from_list(dict_dataset)
    return new_dataset
        

def split_long_and_short(json_file, dataset):
    
    math_str = "Please reason step by step, and put your final answer within \\boxed{}"
    
    def split_by_length(example):
        text_length = len(tokenizer.tokenize(example['instruction'] + example['output'])) # 替换"text"为您的文本字段名
        example['is_long'] = text_length >= 32768
        example['is_math'] = math_str in example['instruction']
        return example
    
    # 添加长度标记
    dataset = dataset.map(split_by_length, num_proc=64)
    
    long_samples = dataset.filter(lambda x: x["is_long"])
    short_samples = dataset.filter(lambda x: not x["is_long"])
    print("samples longer than 32K: ", len(long_samples))
    print("samples shorter than 32K: ", len(short_samples))

    long_sample_file = json_file[:-6] + "_gt_32K.jsonl"
    short_sample_file = json_file[:-6] + "_lt_32K.jsonl"
    print(f"Long examples are saved to {long_sample_file}")
    print(f"Short examples are saved to {short_sample_file}")
    long_samples.to_json(long_sample_file)
    short_samples.to_json(short_sample_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model path or name used to load tokenizer
    parser.add_argument('--model_name', type=str, required=False, Default="Qwen/Qwen2.5-32B") 
    # Contamination source data (e.g., "math_sample1.jsonl" "math_sample1.jsonl")
    parser.add_argument('--json_file_paths', nargs='+', type=str, required=True, help='split by space when input more than one path')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    json_files = args.json_file_paths
    
    for json_file in json_files:
        # convert original dataset to jsonl format
        dataset = convert_json(json_file)
        
        if "math" in json_file:
            prompt = user_prompt_math
        else:
            prompt = user_prompt
        # apply chat prompt
        def mapping(sample, prompt):
            sample["instruction"] = system + prompt.replace("{{content}}", sample["instruction"])
            sample["output"] = sample["output"] + "<|im_end|>"
            return sample
        print("Construct chat template for each sample ...")
        dataset = dataset.map(mapping, num_proc=64, fn_kwargs={"prompt": prompt})
        
        # split long and short cot samples
        split_long_and_short(json_file, dataset)


# Copyright 2025 PCL and Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" for AM-DeepSeek-R1-0528-Distilled """
import numpy as np
from transformers import AutoTokenizer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.dataset.handler.base_handler import BaseInstructDataHandler


MAX_TOKEN_LENGTH = 32768

max_length = 0
length_larger_than_40k_count = 0

@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class AMDeepSeekDataHandler(BaseInstructDataHandler):
    """AM DeepSeek Data Handler"""

    def format_func(self, example):
        """format jsonl data as messages"""
        messages = example.get("instruction", "") + example.get("output", "") + "\n"

        return messages

    def tokenize_func(self, messages):
        """tokenize func"""
        # encode messages
        input_ids = self.tokenizer.encode(messages)

        # get the start token id of assistant's answer
        target_index = 0
        for index in range(len(input_ids)):
            if input_ids[index] == 151644 and input_ids[index+1] == 77091:
                target_index = index + 3
                break

        # record maximum length among all data
        global max_length
        if len(input_ids) > max_length:
            max_length = len(input_ids)
            print(f"max_length: {max_length}", flush=True)

        # truncate the data if its length is greater than 32k
        if len(input_ids) > MAX_TOKEN_LENGTH:
            global length_larger_than_40k_count
            length_larger_than_40k_count += 1
            print(f"clip count: {length_larger_than_40k_count}", flush=True)
            input_ids = input_ids[:MAX_TOKEN_LENGTH] + input_ids[-2:len(input_ids)]

        # tokenize the responses of assistant as labels, and set the system's and user's parts to ignore_token_id(default :-100)
        # then merged into a sequence of the same length as input_ids
        labels = input_ids[target_index:]
        ignore_length = target_index
        labels = np.concatenate([np.full(ignore_length, self.ignore_token_id), labels])
        assert len(labels) == len(input_ids), f"input_ids length {len(input_ids)} different from labels {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels.tolist(),
        }

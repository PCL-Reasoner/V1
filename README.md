# ​**PCL-Reasoner-V1**​

## Model Overview  
We release ​**PCL-Reasoner-V1**​, a model trained based on ​**Qwen2.5-32B-Base**​ and undergoes high-performance supervised fine-tuning based on the ​**MindSpore framework**​ and ​**Ascend hardware**. After fine-tuning, the model demonstrates significant improvements in mathematical reasoning capabilities. PCL-Reasoner-V1 achieves 85.7% and 84.2% respectively on AIME 24 and AIME 25, which position PCL-Reasoner-V1 among the top-tier models in the 32B parameter class.  

We have fully open-sourced the model weights, dataset and training code. Follow the tutorial below to deploy and explore post-training!  

![eval_results](images/README/eval_results.png)  

## Development Guide  

### 1. Model Files  
PCL-Reasoner-V1 is fine-tuned from Qwen2.5-32B-Base using MindFormers. Key files include:  

​**Data Processing:​**​  
```
pcl_reasoner_v1
  ├── qwen2_5_tokenizer.py        # qwen2_5 tokenizer
  ├── packing_handler.py          # Data packing process
  └── data_preprocess                        
  	├── decontaminate.py          # validation set contamination detection
  	└── dataset_prehandle_and_split.py # dataset prehandle and split
```

​**Model Configuration:​**​  
```
pcl_reasoner_v1/config
  ├── data_process_handling.yaml           # Format conversion configuration file
  ├── data_process_packing.yaml            # Data packing configuration file
  └── finetune_pcl_reasoner_v1_32k.yaml  # Model fine-tuning configuration file
```

​**Task Launch Script:​**​  
```
pcl_reasoner_v1
  └── run_pcl_reasoner_v1_finetune.sh  # Model fine-tuning launch script
```


### 2. Environment & Data Setup  
#### 2.1 Environment Installation  
| Software | Version |  
|----------|---------|  
| Firmware & Driver | 24.1.rc3.5 |  
| CANN | 7.7.T9.0.B057:8.1.RC1 |  
| Python | 3.10 |  
| MindSpore | 2.6.0 |  
| MindSpore TransFormers | r1.5.0 |

#### 2.2 Data Processing

##### 2.2.1 Dataset Download

Users can download the original dataset from HuggingFace:

| Dataset Name                    | Dataset Link                                                                                                                  |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| AM-DeepSeek-R1-0528-Distilled | [https://huggingface.co/a-m-team/AM-DeepSeek-R1-0528-Distilled](https://huggingface.co/a-m-team/AM-DeepSeek-R1-0528-Distilled) |

##### 2.2.2 Data Preprocessing

First, we perform detection and filtering on the source data through two steps: ​​validation set contamination detection​​ and ​​data filtering​​.

* Validation Set Contamination Detection​​: We use the ​​all-MiniLM-L6-v2​​ model to calculate text cosine similarity and detect contamination in the original mathematical data against the AIME24/25 evaluation set. 
After execution, the script prints detection results in the terminal and saves questions with similarity exceeding the threshold (along with matched evaluation questions) to the specified output path.

  ```
  python PCL-Reasoner-V1/pcl_reasoner_v1/data_preprocess/decontaminate.py \
  --target_data /path/to/target_data \
  --contaminant_source PCL-Reasoner-V1/pcl_reasoner_v1/data_preprocess/aime2425_questions.json \
  --model_path /path/to/distilled/model_path \
  --output_file_prefix /path/to/output_file_prefix
  --threshold 0.7

  # Parameter Description
  target_data: Data to be detected
  contaminant_source: Contamination source (evaluation set data)
  model_path: Model for text embedding calculation
  output_file_prefix: Output path for results
  threshold: Similarity threshold
  ```
* Data Filtering & Processing​​: Execute the data processing script to filter data by length, selecting data where the combined length of the question and reasoning chain is ​​<32K tokens​​, and add prompts to the data.

  ```
  python PCL-Reasoner-V1/pcl_reasoner_v1/data_preprocess/convert_and_split_dataset.py \
  --json_file_paths /path/to/AM-DeepSeek-R1-0528-Distilled/math.jsonl

  # Parameter Description
  json_file_paths: Dataset to process (multiple paths separated by spaces)
  ```

Then, we convert data into ​​packed format​​ through two sequential steps: format conversion and data packing.

* Format Conversion​​: Specify paths like `data_files`、`vocab_file`、`merges_file` in `pcl_reasoner_v1/config/data_process_handling.yaml`, specify the custom AMDeepSeekDataHandler from pcl_reasoner_v1/packing_handler.py as the data handler:

  ```
  train_dataset:
      ...
      path: "json" # Original dataset format
      data_files:
          ["/path/to/data.jsonl"] # Path to raw dataset
      input_columns: *input_columns
      handler:
        - type: AMDeepSeekDataHandler # Custom data handler class
          ...
          tokenizer:
            auto_register: qwen2_5_tokenizer.Qwen2Tokenizer
            ...
            vocab_file: "/path/to/vocab.json" # Qwen2.5 tokenizer vocabulary
            merges_file: "/path/to/merges.txt" # Qwen2.5 merge rules
            ...
  ```

  *(Note: This is a minimal example showing frequently modified fields. Full configuration is available in the code repository.)*

  Execute the conversion script to generate ​​Arrow-format data​​:

  ```
  export PYTHONPATH=/path/to/mindformers/:PYTHONPATH
  python th/to/mindformers/toolkit/data_preprocess/huggingface/datasets_preprocess.py
      --config ./pcl_reasoner_v1/config/data_process_handling.yaml
      --save_path /path/to/handled_data/
      --register_path ./pcl_reasoner_v1/

  # Parameter Description
  config: Path to format conversion config file
  save_path: Output directory for processed data
  register_path: Path to custom handler registration
  ```
* Data Packing​​:

  Configure pcl_reasoner_v1/config/data_process_packing.yaml to specify input paths for packed data generation:

  ```
  # dataset
  train_dataset:
    data_loader:
    ...
    path: /path/to/handled_data # Processed dataset
    ...
  ```

  *(Note: Example shows key fields only. Refer to repository for full config.)*

  Run the packing script to generate ​​sequence-packed data​:

  ```
  export PYTHONPATH=/path/to/mindformers/:PYTHONPATH
  python /path/to/mindformers/toolkit/data_preprocess/huggingface/datasets_preprocess.py
      --config ./pcl_reasoner_v1_config/data_process_packing.yaml
      --save_path /path/to/packed_data/
      --register_path ./pcl_reaoner_v1/

  # Parameter Description
  config: Path to data packing config file
  save_path: Output directory for packed data
  register_path: Path to custom handler registration
  ```


### 3 Training Process
#### 3.1 Weight Preparation

Users can download pre-trained weights from HuggingFace:

| Model Name          | Weights URL                                                                        |
| ------------------- | --------------------------------------------------------------------------------- |
| Qwen2.5-32B-Base | [https://huggingface.co/Qwen/Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B) |

​**Note**: MindFormers v1.5.0+ supports direct loading/saving of `safetensors` format weights. No conversion to `ckpt` is required. Subsequent fine-tuning will use `safetensors` format.

#### 3.2 Training Configuration
*(Only frequently modified configurations are shown. Full config: `pcl_reasoner_v1/config/finetune_pcl_reasoner_v1_32k.yaml`)*

​***Basic Configuration:​**​*
```yaml
run_mode: 'finetune'               # Training mode: fine-tuning
load_checkpoint: '/path/to/Qwen-32B-base/'  # Weight file path
load_ckpt_format: 'safetensors'    # Weight format
auto_trans_ckpt: True              # Enable online weight splitting for distributed training
```
***Dataset Configuration:​***  
```yaml
train_dataset: &train_dataset
  data_loader:
    type: CommonDataLoader
    path: "/path/to/dataset/pack_data_lt_32K_full"  # Packed dataset path
    load_func: 'load_from_disk'     # Data loading method
    shuffle: True                  # Enable data shuffling
    packing: pack                  # Packed data format
    adaptor_config:
      compress_mask: True
    mock_config:
      seq_length: 32768            # Packed sequence length (32K tokens)
      size: 25909                  # Dataset size / data parallelism split
```
***Parallelism Configuration:​***
```yaml
parallel_config:
  data_parallel: &dp 8             # Data parallelism
  model_parallel: 8                # Model parallelism
  pipeline_stage: 2                # Pipeline parallelism stages
  use_seq_parallel: True           # Enable sequence parallelism
  optimizer_shard: True            # Enable optimizer sharding
  micro_batch_num: 16              # Micro-batch size
```
> *（Note: This configuration example only lists frequently modified items. Refer to the code repository for complete configurations.）*

#### 3.3 Launching Fine-tuning

Specify the configuration file `pcl_reasoner_v1/config/finetune_pcl_reasoner_v1_32k.yaml` in the launch script `run_pcl_reasoner_v1_finetune.sh`, and modify cluster parameters according to your hardware environment:

```bash
noderank=$1 

bash /path/to/mindformers/scripts/msrun_launcher.sh "run_mindformer.py \
--config /path/to/finetune_pcl_reasoner_v1_32k.yaml \
--run_mode finetune" \
--worker_num 128 \
--local_worker_num 8 \
--master_addr XX.XX.XX.XX \
--master_port XXXX \
--node_rank $noderank \
--log_dir /path/to/log \
--join False \
--cluster_time_out 1200 \
> run.log 2>&1

# Parameter Description
config: Path to configuration file
run_mode: Operation mode (pretrain/finetune/inference)
worker_num: Total number of accelerator cards
local_worker_num: Cards per single server
master_addr: Master node address
master_port: Master node port
log_dir: Log directory path
join: Whether to wait for all workers to exit
cluster_time_out: Cluster timeout duration
```
Then, launch the fine-tuning task using:
```
bash run_pcl_reasoner_v1_finetune.sh 0
```
> *（Note: When launching on multiple nodes, specify node_rank (e.g., 0 for the first node).）*

After starting the task, monitor the runtime logs with:
```
tail -f /path/to/log/worker_127.log
```
      
### 4. Evaluation 

To ensure the fairness of evaluation results, we adopted the ​**open-source evaluation code from QwQ**​ ([QwQ/eval at main · QwenLM/QwQ](https://github.com/QwenLM/QwQ)). Developers can follow the `README.md` in the code repository to set up the environment and evaluate models.  

#### Evaluation Hyperparameters  
The sampling hyperparameters used are listed below:  

| Hyperparameter | Value |  
|----------------|---------------------------------|  
| `temperature`  | 0.6                             |  
| `top_k`        | 40                              |  
| `top_p`        | 0.95                            |  
| `max_tokens`   | 129,024                         |  
| `chat_template`| `./pcl_reasoner_v1/eval/am_thinking.jinja` |  

#### Evaluation Results on AIME24/25  
The table below compares mainstream models on the AIME24 and AIME25 benchmarks. For accuracy, we used the ​**Avg@32 metric**​ (averaging 32 sampling attempts per query):  



<table>
  <tr>
    <th>Parameter Size</th>
    <th>Model Name</th>
    <th>AIME 24</th>
    <th>AIME 25</th>
  </tr>
  <!-- 合并行表头 >100B -->
  <tr>
    <th rowspan="6">&gt;100B</th>
  </tr>
  <!-- >100B 组数据行 -->
  <tr>
    <td>DeepSeek-R1</td>
    <td><span style="color:grey">79.8</span></td>
    <td><span style="color:grey">70</span></td>
  </tr>
  <tr>
    <td>DeepSeek-R1-0528</td>
    <td><span style="color:red">91.4</span></td>
    <td><span style="color:red">87.5</span></td>
  </tr>
  <tr>
    <td>Qwen3-235B-A22B</td>
    <td><span style="color:grey">85.7</span></td>
    <td><span style="color:grey">81.5</span></td>
  </tr>
  <tr>
    <td>OpenAI-o3</td>
    <td><b>91.6</b></td>
    <td><b>88.9</b></td>
  </tr>
  <tr>
    <td>Gemini-2.5-Pro-0506</td>
    <td><span style="color:red">90.8</span></td>
    <td><span style="color:grey">83</span></td>
  </tr>
  <!-- 合并行表头 32B -->
  <tr>
    <th rowspan="7">32B</th>
  </tr>
  <!-- 32B 组数据行 -->
  <tr>
    <td>Qwen3-32B</td>
    <td><span style="color:grey">81.4</span></td>
    <td><span style="color:grey">72.9</span></td>
  </tr>
  <tr>
    <td>QwQ-32B</td>
    <td><span style="color:grey">79.5</span></td> 
    <td><span style="color:grey">69.5</span></td>
  </tr>
  <tr>
    <td>DeepSeek-R1-Distill-Qwen-32B</td>
    <td><span style="color:grey">72.6</span></td>
    <td><span style="color:grey">49.6</span></td> 
  </tr>
  <tr>
    <td>Skywork-OR1-32B</td>
    <td><span style="color:grey">82.2</span></td>
    <td><span style="color:grey">73.3</span></td>
  </tr>
  <tr>
    <td>AM-Thinking-v1</td>
    <td><span style="color:grey">85.3</span></td>
    <td><span style="color:grey">74.4</span></td>
  </tr>
  <tr>
    <td>PCL-Reasoner-v1</td>
    <td><b>85.7</b></td>
    <td><b>84.2</b></td>
  </tr>
</table>

> *Note: Generated results for AIME24/25 are available in the [`pcl_reasoner_v1/eval/eval_res`](https://openi.pcl.ac.cn/PCL-Reasoner/V1) directory for developer verification and comparison.*  

#### Impact of Answer Length on Accuracy  
We analyzed the relationship between maximum answer length (`max_tokens`) and model accuracy. Due to results listed below, we find that on AIME24 which is relatively simpler, decode length of 64K​ are sufficient to achieve peak accuracy of 85.7%. In contrast, AIME25 which is relatively harder requires ​128K tokens​ to reach optimal performance (84.2%):

<table>
  <tr>
    <th>max tokens</th>
    <th>16K</th>
    <th>32K</th>
    <th>64K</th>
    <th>128K</th>
  </tr>
  <tr>
    <td>AIME24</td>
    <td>42.0</td>
    <td>77.9</td>
    <td>85.7</td>
    <td>85.7</td>
  </tr>
  <tr>
    <td>AIME25</td>
    <td>33.4</td>
    <td>75.6</td>
    <td>83.9</td>
    <td>84.2</td>
  </tr>
</table>

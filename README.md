
# Flatness: Run Guide

This guide matches your **full run commands**. All paths are derived from `--prefix=/your_path`; the repo root is **`/your_path/Flatness`**. The end-to-end flow is: Environment → Data gen → Training (metric vs random) → **Evaluation** → Calculate results.

-----

## 0\. Prerequisites

  * Conda
  * At least 1 GPU
  * Directory layout (after running, you’ll have something like this):
    ```text
    /your_path
    ├─ Flatness/                          # repo root (this doc lives here)
    │   ├─ eagle/                         # source
    │   ├─ result.py                      # final aggregation script
    │   └─ ...
    ├─ pretrained_models/llama-3-8b-instruct/
    ├─ data/
    │   └─ eagle/data/1/                  # generated hidden states
    ├─ output/ckp/
    │   ├─ run_metric/                    # "metric" training outputs
    │   └─ run_random/                    # "random" training outputs
    └─ Flatness/gsm8k/
        ├─ ea_metric-temperature-1.0.jsonl
        └─ baseline_random-temperature-baseline-1.0.jsonl
    ```

> **Note:** All paths are derived from `--prefix`.

-----

## 1\. Environment Setup

```bash
cd /your_path/Flatness
conda create -y --name eagle python=3.10
conda activate eagle
pip install --upgrade pip
pip install -r requirements.txt
python download_model.py
```

  * `download_model.py` should place the base model at: `/your_path/pretrained_models/llama-3-8b-instruct/` and hf tokens

-----

## 2\. Hidden-State Generation (for training)

This step now accepts a path to a specific JSON file for data generation.

```bash
conda activate eagle
cd /your_path/Flatness/eagle/ge_data

# Example: use GPUs 0,1; process data from a specified JSON file (your_dataset.json)
# Generate range [0, 68622) (use sharegpt as an example) into {prefix}/data/eagle/data/1
python ge_data_all_llama3.py \
  --prefix /your_path \
  --dataset_json_path /your_path/Flatness/dataset/your_dataset.json \
  --start 0 \
  --end 68622 \
  --index 1 \
  --gpu_index 0 1
```

**Outputs:** `/your_path/data/eagle/data/1/data_*.ckpt`

-----

## 3\. Training (two tracks)

> **Note:** In our data filtering procedure, a sample refers to an entire **batch**, since training iterates by batches rather than individual items.


### A. Flatness-based metric (index=1; keep 50%)


```bash
cd /your_path/Flatness/eagle/train
deepspeed --num_gpus=8 main_deepspeed_index.py \
  --deepspeed --deepspeed_config /your_path/ds_config.json \
  --prefix /your_path \
  --basepath /your_path/pretrained_models/llama-3-8b-instruct \
  --tmpdir /your_path/data/eagle/data/1 \
  --cpdir /your_path/output/ckp/run_metric \
  --epochs 200 \
  --time_out_path /your_path/Flatness/result/total_training_time_metric.csv \
  --filter_start_epoch 1 \
  --quantile 0.5 \
  --index 1
```

### B. Random baseline (index=0; keep 50%)

```bash
cd /your_path/Flatness/eagle/train
deepspeed --num_gpus=8 main_deepspeed_index.py \
  --deepspeed --deepspeed_config /your_path/ds_config.json \
  --prefix /your_path \
  --basepath /your_path/pretrained_models/llama-3-8b-instruct \
  --tmpdir /your_path/data/eagle/data/1 \
  --cpdir /your_path/output/ckp/run_random \
  --epochs 200 \
  --time_out_path /your_path/Flatness/result/total_training_time_random.csv \
  --filter_start_epoch 1 \
  --quantile 0.5 \
  --index 0
```

-----

## 4\. Evaluation (EA vs Baseline)

Benchmark names (`--bench-name`) can include `gsm8k`, `alpaca`, `mt_bench`, `qa`, `sum`.

### EA evaluation (temperature = 1.0)

```bash
cd /your_path/Flatness/eagle/evaluation
CUDA_VISIBLE_DEVICES=0 python3 gen_ea_answer_llama3chat.py \
  --prefix /your_path \
  --ea-model-path   /your_path/output/ckp/run_metric/state_199 \
  --base-model-path /your_path/pretrained_models/llama-3-8b-instruct \
  --bench-name gsm8k \
  --model-id ea_metric \
  --temperature 1.0
```

### Baseline evaluation (temperature = 1.0)

```bash
cd /your_path/Flatness/eagle/evaluation
CUDA_VISIBLE_DEVICES=0 python3 gen_baseline_answer_llama3chat.py \
  --prefix /your_path \
  --ea-model-path   /your_path/output/ckp/run_metric/state_199 \
  --base-model-path /your_path/pretrained_models/llama-3-8b-instruct \
  --bench-name gsm8k \
  --model-id baseline_metric \
  --temperature 1.0
```

**Default outputs:**

  * **EA:** `/your_path/Flatness/gsm8k/ea_metric-temperature-1.0.jsonl`
  * **Baseline:** `/your_path/Flatness/gsm8k/baseline_random-temperature-baseline-1.0.jsonl`

> To customize the output file/directory, use the `--answer-file` argument.

-----

## 5\. Calculate results

```bash
cd /your_path/Flatness
python result.py \
  --prefix /your_path \
  --bench-name gsm8k \
  --model_ids ea_metric,baseline_metric \
  --temperature 1.0
```

**Output:** `/your_path/Flatness/result/summary_temperature_1.0.csv`


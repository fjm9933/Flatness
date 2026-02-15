# ge_data_all_llama3.py  -- sanitized & prefix-based
# -*- coding: utf-8 -*-
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Generate teacher hidden states & masks (prefix-based, no hardcoded paths)")
parser.add_argument('--prefix', type=str, required=True,
                    help='Project root prefix, e.g., /your_path')
parser.add_argument('--dataset_json_path', type=str, required=True,
                    help='Path to the input dataset JSON file.')
parser.add_argument('--start', type=int, required=True,
                    help='Start index in dataset (inclusive)')
parser.add_argument('--end', type=int, required=True,
                    help='End index in dataset (exclusive)')
parser.add_argument('--index', type=int, required=True,
                    help='Output subdir index tag, e.g., 1')
parser.add_argument('--gpu_index', type=int, nargs='+', required=True,
                    help='CUDA device indices, e.g., 0 1 2 3 or just 0')
args = parser.parse_args()

# ---- Derive all paths from --prefix (no hardcoded paths) ----
prefix = args.prefix.rstrip('/')
bigname = f"{prefix}/pretrained_models/llama-3-8b-instruct"      # teacher model folder
outdir_root = f"{prefix}/data/eagle/data"                         # output root
outdir = f"{outdir_root}/{args.index}"

# ---- Devices ----
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_index))

# ---- Load teacher model/tokenizer ----
bigmodel = AutoModelForCausalLM.from_pretrained(bigname, device_map="auto", torch_dtype=torch.float16)
bigtokenizer = AutoTokenizer.from_pretrained(bigname, use_fast=False)
bigmodel.eval()

def build_dataset_rank(tokenizer):
    ds = load_dataset('json', data_files=args.dataset_json_path)['train']
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(args.start, min(args.end, len(ds))))
    original_columns = ds.column_names

    def preprocess_function(examples):
        new_examples = {
            "conversation": [],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            messages = [
                {"role": "system",
                 "content":
                 "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
                 "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, "
                 "dangerous, or illegal content. Ensure responses are unbiased and positive. "
                 "If a question is nonsensical or factually incoherent, explain why. If you don't know, say so."},
            ]
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples['conversations'][i]
            if roles[source[0]["from"]] != "user":
                source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                if sentence["from"] == "gpt":
                    sentence["value"] = " " + sentence["value"]
                messages.append({"role": role, "content": sentence["value"]})

            conversation = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # tokenizer pad fix
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                add_special_tokens=False,
            ).input_ids[0]

            loss_mask = torch.ones_like(input_ids)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"

            turns = conversation.split(sep2)
            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for k, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # llama offset -1

                if k == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if k != 0:
                    cur_len += 3

            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        remove_columns=original_columns,
        load_from_cache_file=False
    )
    ds.set_format(type="torch")
    return ds

@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    td = {
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0]
    }
    return td

def writedata(dirpath, data_point):
    os.makedirs(dirpath, exist_ok=True)
    idx = len(os.listdir(dirpath))
    torch.save(data_point, f"{dirpath}/data_{idx}.ckpt")

def main():
    os.makedirs(outdir, exist_ok=True)
    ds = build_dataset_rank(bigtokenizer)
    print(ds)
    for i, data in enumerate(ds):
        if i % 100 == 0:
            print(i, end="\t")
        if i % 1000 == 0:
            print("")
        outdata = ge(data)
        writedata(outdir, outdata)

if __name__ == "__main__":
    main()

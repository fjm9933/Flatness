# -*- coding: utf-8 -*-
"""
Generate answers with local baseline model (sanitized: no hardcoded paths).

Usage example:
CUDA_VISIBLE_DEVICES=0 python3 -m eagle.evaluation.gen_baseline_answer_llama3chat \
  --prefix /your_path \
  --ea-model-path /your_path/output/ckp/run_random/state_10 \
  --base-model-path /your_path/pretrained_models/llama-3-8b-instruct \
  --bench-name gsm8k \
  --model-id baseline_random_state10 \
  --temperature 0.0
"""
import argparse
import json
import os
import time
import torch
import shortuuid
from accelerate.utils import set_seed
set_seed(0)

from tqdm import tqdm
from fastchat.llm_judge.common import load_questions

from ..model.ea_model import EaModel
from ..model.kv_cache import initialize_past_key_values  # noqa: F401
from ..model.utils import prepare_logits_processor  # noqa: F401


def ensure_fresh_answer_file(path: str) -> str:
    """If an output file already exists, remove it, then ensure its parent dir exists."""
    path = os.path.expanduser(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"[clean] removed existing file: {path}")
        except Exception as e:
            raise RuntimeError(f"Cannot remove existing output file '{path}': {e}")
    return path


def run_eval(
    base_model_path,
    ea_model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    args
):
    questions = load_questions(question_file, question_begin, question_end)

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    workers = max(1, (num_gpus_total // num_gpus_per_model))
    chunk_size = max(1, len(questions) // workers)

    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        import ray
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    base_model_path,
    ea_model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    args
):
    # Baseline: use EaModel.naivegenerate for standard decoding
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        threshold=getattr(args, "threshold", 0.1),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()
    logits_processor = prepare_logits_processor(temperature=temperature) if temperature > 1e-5 else None

    model.eval()
    print('Check model training state:', model.training)
    print('CUDA VISIBLE DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))

    # Warmup on first question if available
    if len(questions) > 0:
        q0 = questions[0]
        for _ in range(3):
            torch.manual_seed(0)
            messages = [{"role": "system",
                         "content": "You are a helpful, respectful and honest assistant. Always answer as "
                                    "helpfully as possible, while being safe. Your answers should not include "
                                    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                                    "Ensure responses are unbiased and positive. If a question is nonsensical or "
                                    "factually incoherent, explain why. If you don't know, say so."}]
            for j in range(len(q0["turns"])):
                qs = q0["turns"][j]
                messages.append({"role": "user", "content": qs})
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

                torch.cuda.synchronize()
                t0 = time.time()
                output_ids, new_token, idx = model.naivegenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                )
                torch.cuda.synchronize()
                _ = time.time() - t0
    print('Warmup done' if len(questions) > 0 else 'No warmup (empty questions)')

    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [{"role": "system",
                         "content": "You are a helpful, respectful and honest assistant. Always answer as "
                                    "helpfully as possible, while being safe. Your answers should not include "
                                    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
                                    "Ensure responses are unbiased and positive. If a question is nonsensical or "
                                    "factually incoherent, explain why. If you don't know, say so."}]
            turns = []; idxs = []; new_tokens = []; wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({"role": "user", "content": qs})
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

                torch.cuda.synchronize()
                t0 = time.time()
                output_ids, new_token, idx = model.naivegenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                )
                torch.cuda.synchronize()
                dt = time.time() - t0

                out_ids = output_ids[0][len(input_ids[0]):]
                stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                if stop_token_ids:
                    stop_idx = [k for k, tid in enumerate(out_ids) if tid in stop_token_ids]
                    if len(stop_idx) > 0:
                        out_ids = out_ids[:stop_idx[0]]

                output = tokenizer.decode(out_ids, spaces_between_special_tokens=False)
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for st in special_token: output = output.replace(st, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(dt)
                messages.append({"role": "assistant", "content": output})
            choices.append({"index": i, "turns": turns, "idxs": idxs,
                            "new_tokens": new_tokens, "wall_time": wall_time})

        os.makedirs(os.path.dirname(answer_file) or ".", exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l
    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-model-path",   type=str, required=True, help="Local folder of EA weights")
    parser.add_argument("--base-model-path", type=str, required=True, help="Local folder of base model")
    parser.add_argument("--bench-name",      type=str, required=True, help="Benchmark name (e.g., gsm8k)")
    parser.add_argument("--model-id",        type=str, required=True, help="Model ID tag for outputs")

    parser.add_argument("--question-file",   type=str, help="Override path to question.jsonl")
    parser.add_argument("--prefix",          type=str, help="Project root to derive question.jsonl, e.g., /your_path")

    parser.add_argument("--answer-file",   type=str, help="Output file; default: {bench-name}/{model-id}.jsonl")
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token",   type=int, default=63)
    parser.add_argument("--depth",         type=int, default=5)
    parser.add_argument("--top-k",         type=int, default=8)
    parser.add_argument("--threshold",     type=float, default=0.1)
    parser.add_argument("--num-choices",   type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total",     type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str)
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end",   type=int)

    args = parser.parse_args()
    args.model_id = args.model_id + "-temperature-baseline-" + str(args.temperature)

    if args.question_file:
        question_file = args.question_file
    elif args.prefix:
        # NOTE: Flatness as repo root when using --prefix
        question_file = f"{args.prefix.rstrip('/')}/Flatness/eagle/data/{args.bench_name}/question.jsonl"
    else:
        # fallback: relative to package
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"

    answer_file = args.answer_file or f"{args.bench_name}/{args.model_id}.jsonl"
    answer_file = ensure_fresh_answer_file(answer_file)
    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.max_gpu_memory,
        args.temperature,
        args
    )
    reorg_answer_file(answer_file)

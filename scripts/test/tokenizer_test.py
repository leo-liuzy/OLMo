import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import datasets as ds
from transformers import AutoTokenizer

def olmo_encode(text, tokenizer, max_seq_len):
    input_ids = [tokenizer.eos_token_id]
    text = text.strip() + tokenizer.eos_token + "\n"
    content_tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids.extend(content_tokens)
    input_ids = input_ids[:max_seq_len]
    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids.extend([tokenizer.pad_token_id] * pad_len)
    return input_ids

def compare_tokenized_data(raw_dataset_path, tokenized_input_ids_file, hf_tokenizer_path, max_seq_len, num_examples_to_check=10):
    raw_ds = ds.load_dataset("json", data_files=raw_dataset_path, split="train")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path)
    memmap_ids = np.memmap(tokenized_input_ids_file, dtype=np.uint16, mode='r')
    num_examples = memmap_ids.shape[0] // max_seq_len
    memmap_ids = memmap_ids.reshape((num_examples, max_seq_len))
    available = min(len(raw_ds), num_examples, num_examples_to_check)
    for idx in range(available):
        text = raw_ds[idx]["text"]
        hf_ids = olmo_encode(text, hf_tokenizer, max_seq_len)
        olmo_ids = memmap_ids[idx].tolist()
        print(f"Example {idx}:")
        print(text[:100] + ("..." if len(text) > 100 else ""))
        print("HF tokens:   ", hf_ids[:10], "...")
        print("OLMo tokens: ", olmo_ids[:10], "...")
        if hf_ids == olmo_ids:
            print("[OK] Tokens match.")
        else:
            print("[WARNING] Tokens do not match.")
            for pos, (hf_id, olmo_id) in enumerate(zip(hf_ids, olmo_ids)):
                if hf_id != olmo_id:
                    print(f"Mismatch at pos {pos}: HF {hf_id}, OLMo {olmo_id}")
                    try:
                        print("HF decoded:", hf_tokenizer.decode([hf_id]))
                    except Exception as e:
                        print("Decoding error:", e)
                    break
        print("=" * 40)

if __name__ == "__main__":
    for split in ["valid", "test", "train"]:
        raw_dataset_path = f"/scratch/07144/yw23374/data/structured_cpt_{split}.jsonl"
        tokenized_input_ids_file = f"/scratch/07144/yw23374/data/astro_cpt_dscoder/{split}/input_ids.npy"
        hf_tokenizer_model = "/scratch/07144/yw23374/base_models/deepseek/hf/deepseek-coder-1.3b-base/"
        max_seq_len = 8192
        compare_tokenized_data(raw_dataset_path, tokenized_input_ids_file, hf_tokenizer_model, max_seq_len, num_examples_to_check=5)
        print(f"=====check {split} done!!======")

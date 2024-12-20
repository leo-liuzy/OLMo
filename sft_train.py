
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

parser = HfArgumentParser((SFTConfig,))
args, = parser.parse_args_into_dataclasses()
model_name_or_path = "/home1/09636/zyliu/scratch/base_models/deepseek/hf/deepseek-coder-1.3b-base"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.padding_side = 'right'
tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

train_dataset = load_dataset("json", data_files="/home1/09636/zyliu/work/OLMo/olmo_data/astro_sft/astro_sft_train.jsonl", split="train")
valid_dataset = load_dataset("json", data_files="/home1/09636/zyliu/work/OLMo/olmo_data/astro_sft/astro_sft_valid.jsonl", split="train")


response_template = "### Response:\n"

model.config.max_position_embeddings

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=args,
    data_collator=collator,
) # type: ignore

trainer.train()
trainer.model.save_pretrained(save_directory=args.output_dir)

trainer.accelerator.wait_for_everyone()



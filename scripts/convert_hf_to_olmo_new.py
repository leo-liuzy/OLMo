# Copyright 2024 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
import numpy as np
from tokenizers import Tokenizer
from transformers import OlmoConfig, OlmoForCausalLM
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from collections import OrderedDict
from olmo.config import TrainConfig, LayerNormType, TokenizerConfig
from olmo.tokenizer import Tokenizer
from tokenizers import Tokenizer as BaseTokenizer
from olmo.model import OLMo
from olmo.checkpoint import FullCheckpointer

"""
Sample usage:
```
python src/transformers/models/olmo/convert_olmo_weights_to_hf.py \
    --input_dir /path/to/downloaded/olmo/weights --model_size 7B --output_dir /output/path
```
Thereafter, models can be loaded via:
```py
from transformers import OlmoForCausalLM, AutoTokenizer
model = OlmoForCausalLM.from_pretrained("/output/path")
tokenizer = AutoTokenizer.from_pretrained("/output/path")
```
Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(
    model_path,
    input_base_path,
    include_tokenizer=True,
    tokenizer_path=None,
    safe_serialization=True,
    fix_eos_token_id=True,
    tmp_cleanup=True,
):
    os.makedirs(model_path, exist_ok=True)
    # tmp_model_path = os.path.join(model_path, "tmp")
    # os.makedirs(tmp_model_path, exist_ok=True)
    
    hf_config = AutoConfig.from_pretrained(input_base_path)
    olmo_config = TrainConfig()
    
    # Convert tokenizer
    tokenizer_path = input_base_path

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    olmo_config.model.pad_token_id = hf_tokenizer.pad_token_id # type: ignore
    olmo_config.model.eos_token_id = hf_tokenizer.eos_token_id # type: ignore
    olmo_config.tokenizer.truncate_direction = hf_tokenizer.truncation_side # type: ignore
    model_name = os.path.basename(tokenizer_path)
    base_tokenizer = BaseTokenizer.from_file(os.path.join(tokenizer_path, 'tokenizer.json'))
    olmo_tokenizer = Tokenizer(
        base_tokenizer, 
        eos_token_id=hf_tokenizer.eos_token_id, # type: ignore
        pad_token_id=hf_tokenizer.pad_token_id,
        truncate_direction=hf_tokenizer.truncation_side
    )
    shutil.copyfile(os.path.join(tokenizer_path, 'tokenizer.json'), f"/home1/09636/zyliu/work/OLMo/olmo_data/tokenizers/{model_name}.json")
    # config_path = Path(input_base_path) / "config.yaml"
    # olmo_config = yaml.safe_load(config_path.read_text())["model"]
    # Set all model setting in olmo's config
    olmo_config.model.d_model = dim = hf_config.hidden_size
    olmo_config.model.n_heads = n_heads = hf_config.num_attention_heads
    
    dims_per_head = dim // n_heads
    
    olmo_config.model.n_kv_heads = num_key_value_heads = hf_config.num_key_value_heads
    olmo_config.model.clip_qkv = None
    olmo_config.model.n_layers = n_layers = hf_config.num_hidden_layers
    olmo_config.model.mlp_hidden_size = hf_config.intermediate_size
    olmo_config.model.activation_type = "swiglu" if hf_config.hidden_act == "silu" else hf_config.hidden_act # type: ignore
    # ! the next line is a hack to overwrite the built-in multiplier in each activation function
    olmo_config.model.activation_output_multiplier = 1 if hf_config.hidden_act == "silu" else None # type: ignore
    olmo_config.model.rope = True
    olmo_config.model.rope_theta = rope_theta = hf_config.rope_theta
    if hf_config.rope_scaling['rope_type'] == 'linear':
        olmo_config.model.rope_factor = hf_config.rope_scaling['factor']
    olmo_config.model.attention_dropout = hf_config.attention_dropout
    # Apply layer norm to the keys and queries within the attention mechanism.
    olmo_config.model.attention_layer_norm = False
    olmo_config.model.attention_layer_norm_with_affine = False
    olmo_config.model.embedding_dropout = 0.
    olmo_config.model.residual_dropout = 0.
    olmo_config.model.embedding_layer_norm = False
    olmo_config.model.layer_norm_type = LayerNormType.rms
    olmo_config.model.layer_norm_with_affine = True
    olmo_config.model.layer_norm_eps = hf_config.rms_norm_eps
    olmo_config.model.max_sequence_length = max_position_embeddings = hf_config.max_position_embeddings
    include_attn_bias = hf_config.attention_bias
    include_mlp_bias = hf_config.mlp_bias
    assert include_attn_bias == include_mlp_bias
    olmo_config.model.include_bias = include_attn_bias
    olmo_config.model.bias_for_layer_norm = False
    olmo_config.model.scale_logits = False
    olmo_config.model.embedding_size = olmo_config.model.vocab_size = vocab_size = hf_config.vocab_size
    olmo_config.model.weight_tying = hf_config.tie_word_embeddings
    olmo_config.model.eos_token_id = hf_config.eos_token_id
    olmo_config.model.pad_token_id = hf_tokenizer.pad_token_id # type: ignore
    
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    
    olmo_config.model.norm_after = False
    
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    
    # Not sharded
    # (The sharded implementation would also work, but this is simpler.)
    # loaded = torch.load("/home1/09636/zyliu/scratch/base_models/OLMo/olmo/OLMo-1B-final/model.pt", map_location="cpu")
    hf_model = AutoModelForCausalLM.from_pretrained(input_base_path)
    
    output_olmo = OrderedDict()
    
    output_olmo["transformer.wte.weight"] = hf_model.model.embed_tokens.weight
    components = ["self_attn.o_proj.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "mlp.up_proj.weight", "mlp.gate_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "model.embed_tokens.weight", "norm.weight"]
    
    
    if not hf_config.tie_word_embeddings:
        output_olmo["transformer.ff_out.weight"] = hf_model.lm_head.weight.data
        components.append("lm_head.weight")
    
    recorded_param_count = 0
    for layer_i, layer in enumerate(hf_model.model.layers):
        # Unsharded
        # TODO: Layernorm stuff
        # TODO: multi query attention
        fused_dims = [dim, dims_per_head * num_key_value_heads, dims_per_head * num_key_value_heads]
        if hasattr(layer, "input_layernorm"):
            output_olmo[f"transformer.blocks.{layer_i}.attn_norm.weight"] = layer.input_layernorm.weight.data
            recorded_param_count += layer.input_layernorm.weight.data.numel()
            if hasattr(layer.input_layernorm, "bias"):
                output_olmo[f"transformer.blocks.{layer_i}.attn_norm.bias"] = layer.input_layernorm.bias.data
                recorded_param_count += layer.input_layernorm.bias.data
            
        output_olmo[f"transformer.blocks.{layer_i}.att_proj.weight"] = \
            torch.cat(
                [
                    layer.self_attn.q_proj.weight.data,
                    layer.self_attn.k_proj.weight.data,
                    layer.self_attn.v_proj.weight.data,
                ]
            )
        # recorded_param_count += output_olmo[f"transformer.blocks.{layer_i}.att_proj.weight"].numel()
        assert output_olmo[f"transformer.blocks.{layer_i}.att_proj.weight"].shape == (sum(fused_dims), dim)
        output_olmo[f"transformer.blocks.{layer_i}.attn_out.weight"] = \
            layer.self_attn.o_proj.weight.data
        # recorded_param_count += output_olmo[f"transformer.blocks.{layer_i}.attn_out.weight"].numel()
        
        if include_attn_bias:
            output_olmo[f"transformer.blocks.{layer_i}.att_proj.bias"] = \
            torch.cat(
                [
                    layer.self_attn.q_proj.bias.data,
                    layer.self_attn.k_proj.bias.data,
                    layer.self_attn.v_proj.bias.data,
                ]
            )
            # recorded_param_count += output_olmo[f"transformer.blocks.{layer_i}.att_proj.bias"].numel()
            
            output_olmo[f"transformer.blocks.{layer_i}.attn_out.bias"] = \
            layer.self_attn.o_proj.bias.data
            # recorded_param_count += output_olmo[f"transformer.blocks.{layer_i}.attn_out.bias"].numel()
        
        if hasattr(layer, "post_attention_layernorm"):
            output_olmo[f"transformer.blocks.{layer_i}.ff_norm.weight"] = layer.post_attention_layernorm.weight.data
            if hasattr(layer.post_attention_layernorm, "bias"):
                output_olmo[f"transformer.blocks.{layer_i}.ff_norm.bias"] = layer.post_attention_layernorm.bias.data
        
        output_olmo[f"transformer.blocks.{layer_i}.ff_proj.weight"] = layer.mlp.up_proj.weight.data
        
        if hasattr(layer.mlp, "gate_proj"):
            olmo_config.model.use_gated_mlp = True
            output_olmo[f"transformer.blocks.{layer_i}.ff_gate.weight"] = layer.mlp.gate_proj.weight.data
        output_olmo[f"transformer.blocks.{layer_i}.ff_out.weight"] = layer.mlp.down_proj.weight.data
        # recorded_param_count += output_olmo[f"transformer.blocks.{layer_i}.ff_proj.weight"].numel()
        # recorded_param_count += output_olmo[f"transformer.blocks.{layer_i}.ff_out.weight"].numel()
        
        if include_mlp_bias:
            output_olmo[f"transformer.blocks.{layer_i}.ff_proj.bias"] = \
            torch.cat(
                [
                    layer.mlp.up_proj.bias,
                    layer.mlp.gate_proj.bias,
                ]
            )
            output_olmo[f"transformer.blocks.{layer_i}.ff_out.bias"] = layer.mlp.down_proj.bias


    output_olmo["transformer.ln_f.weight"] = hf_model.model.norm.weight.data
    if hasattr(hf_model.model.norm, "bias"):
        output_olmo["transformer.ln_f.bias"] = hf_model.model.norm.bias.data
    
    
    # Sanity checking sucess of converting the model
    converted_param_count = 0
    for k, v in output_olmo.items():
        # index_dict["weight_map"][k] = filename
        converted_param_count += v.numel()
    
    model_param_names = [n for n, _ in hf_model.named_parameters()]
    unconverted_model_params = [n for n in model_param_names if not any(x in n for x in components)]
    original_model_params_count = sum([np.prod(p.size()) for p in hf_model.parameters()])
    assert original_model_params_count == converted_param_count, "!!!!! Parameter count mismatch !!!!!"
    assert len(unconverted_model_params) == 0, "!!!!! Unconverted module in hf checkpoint !!!!!\n" + str(unconverted_model_params)
    
    
    olmo_model = OLMo(olmo_config.model)
    olmo_model.load_state_dict(output_olmo)
    
    test_sentence = "Writing checkpoint converter is fun!"
    with torch.no_grad():
        hf_input = hf_tokenizer(test_sentence, return_tensors="pt", add_special_tokens=False)
        hf_output = hf_model(**hf_input)
        
        olmo_input_ids = olmo_tokenizer.encode_batch([test_sentence], add_special_tokens=False)
        assert hf_input['input_ids'].tolist() == olmo_input_ids # type: ignore
        # olmo_input = ol
        # labels = get_labels(hf_input)
        olmo_output = olmo_model(**hf_input)
    torch.save(output_olmo, os.path.join(model_path, "model.pt"))
    olmo_config.save(os.path.join(model_path, "config.yaml"))
    # Make space so we can load the model properly now.
    # del loaded
    # 
    gc.collect()

    
    
    if include_tokenizer:
        _write_tokenizer(model_path, config, input_base_path, tokenizer_path)

def get_labels(batch: Dict[str, Any]) -> torch.Tensor:
    # Labels are just input IDs shifted to the left (first item is ignored).
    labels, label_mask, attention_mask, instance_mask = (
        batch["input_ids"].clone(),
        batch.get("label_mask"),
        batch.get("attention_mask"),
        batch.get("instance_mask"),
    )
    if label_mask is not None:
        labels.masked_fill_(~label_mask, -100)
    if attention_mask is not None:
        labels.masked_fill_(attention_mask == 0.0, -100)
    if instance_mask is not None:
        labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
    return labels[..., 1:].contiguous()


def _write_tokenizer(
    output_path: Path,
    config: OlmoConfig,
    checkpoint_dir: str,
    input_tokenizer_path: Path | None,
) -> None:
    print(f"Saving a {GPTNeoXTokenizerFast.__name__} to {output_path}.")

    if input_tokenizer_path is not None:
        base_tokenizer = Tokenizer.from_file(str(input_tokenizer_path))
    else:
        config_path = Path(checkpoint_dir) / "config.yaml"
        tokenizer_config = yaml.safe_load(config_path.read_text())["tokenizer"]

        # Initialize tokenizer and validate vocab size.
        if Path(tokenizer_config["identifier"]).is_file():
            base_tokenizer = Tokenizer.from_file(tokenizer_config["identifier"])
        else:
            base_tokenizer = Tokenizer.from_pretrained(tokenizer_config["identifier"])

    eos_token_id = config.eos_token_id if config.eos_token_id is not None else base_tokenizer.get_vocab_size() - 1
    pad_token_id = config.pad_token_id if config.pad_token_id is not None else eos_token_id

    tokenizer = GPTNeoXTokenizerFast(
        tokenizer_object=base_tokenizer,
        eos_token=base_tokenizer.decode([eos_token_id], skip_special_tokens=False),
        pad_token=base_tokenizer.decode([pad_token_id], skip_special_tokens=False),
        unk_token=None,
        bos_token=None,
    )

    tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of OLMo weights, which contains config.yaml and model.pt.",
    )
    parser.add_argument(
        "--no_tokenizer",
        action="store_false",
        dest="include_tokenizer",
        help="If set, do not convert OLMo tokenizer to HF tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_json_path",
        type=Path,
        default=None,
        help="Location of OLMo tokenizer json file. Defaults to what is set in the config file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--no_fix_eos_token_id",
        action="store_false",
        dest="fix_eos_token_id",
        help="If set, does not change eos token id from 0 to 50279 if it is 0. Changing 0 to 50279 is a bug fix, so use this option with care.",
    )
    parser.add_argument(
        "--no_tmp_cleanup",
        action="store_false",
        dest="tmp_cleanup",
        help="If passed, don't remove temp dir at end of HF conversion.",
    )
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    # Different OLMo versions used different default values for max_position_embeddings, hence the need to be able to specify which version is being used.
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        include_tokenizer=args.include_tokenizer,
        fix_eos_token_id=args.fix_eos_token_id,
        tmp_cleanup=args.tmp_cleanup,
    )


if __name__ == "__main__":
    main()

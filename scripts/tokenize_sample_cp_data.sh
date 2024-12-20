
PROCESSED_DATA_DIR=${SCRATCH}/processed_data/tenk_dump_package/dedup

# processed_strategy=cat_mds
# processed_strategy=no_biblio_mds
# processed_strategy=no_html_mds
# processed_strategy=no_tab_fig_mds

# for processed_strategy in cat_mds no_biblio_mds no_html_mds no_tab_fig_mds
# do 
#     for split in train
#     do 
#         dolma tokens \
#             --documents ${PROCESSED_DATA_DIR}/${split}/${processed_strategy}/raw.jsonl \
#             --tokenizer.name_or_path allenai/eleuther-ai-gpt-neox-20b-pii-special \
#             --tokenizer.eos_token_id 50279 \
#             --destination ${PROCESSED_DATA_DIR}/${split}/${processed_strategy} \
#             --processes 16 \

#     done
# done

for split in train valid test
do
    echo Processing `$split` data
    python scripts/prepare_tulu_data.py olmo_data/tulu_dscoder_${split} --dataset olmo_data/astro_sft_${split}.jsonl --tokenizer olmo_data/tokenizers/deepseek-coder-1.3b-base.json --eos 32014 --pad 32014
done
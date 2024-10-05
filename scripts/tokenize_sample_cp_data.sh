
PROCESSED_DATA_DIR=${SCRATCH}/processed_data/sample_dump_package/dedup

processed_strategy=cat_mds
# processed_strategy=no_biblio_mds
# processed_strategy=no_html_mds
# processed_strategy=no_tab_fig_mds

for processed_strategy in cat_mds no_biblio_mds no_html_mds no_tab_fig_mds
do 
    for split in train valid
    do 
        dolma tokens \
            --documents ${PROCESSED_DATA_DIR}/${processed_strategy}/${split}/raw.jsonl \
            --tokenizer.name_or_path allenai/eleuther-ai-gpt-neox-20b-pii-special \
            --tokenizer.eos_token_id 50279 \
            --destination ${PROCESSED_DATA_DIR}/${processed_strategy}/${split} \
            --processes 16 \

    done
done
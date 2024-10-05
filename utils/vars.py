import os

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_DATA_DIR = os.path.join(os.environ["SCRATCH"], "data")
PROCESSED_DATA_DIR = os.path.join(os.environ["SCRATCH"], "processed_data")
# OUTPUT_ROOT = os.path.join(PROJ_DIR, "chkpt")
# INITIAL_RESULT_ROOT = os.path.join(OUTPUT_ROOT, "initial")
# INITIAL_RAG_DIR = os.path.join(INITIAL_RESULT_ROOT, "rag")
# OUTPUT_INIT_DIR = os.path.join(OUTPUT_ROOT, "init_response")

# SYS_TEMPLATE_ENTRY = "system_prompt_template"
# USER_TEMPLATE_ENTRY = "user_prompt_template"

def GPT_4_TOKENIZER(x):
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4")
    return enc.encode(x)
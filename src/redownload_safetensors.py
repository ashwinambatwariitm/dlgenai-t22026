# pip install -U huggingface_hub

from huggingface_hub import snapshot_download

# allow_patterns restricts download to safetensors + config/tokenizer files,
# skipping the old pytorch_model.bin entirely
common_patterns = [
    "*.json", "*.txt", "*.model", "*.safetensors",
    "tokenizer*", "spm.model", "sentencepiece.bpe.model"
]

deberta_path = snapshot_download(
    repo_id="microsoft/deberta-v3-small",
    local_dir="/home/studinstru/dlGenAI/dlgenai-t22026/models/deberta-v3-small",
    allow_patterns=common_patterns,
)

roberta_path = snapshot_download(
    repo_id="roberta-base",
    local_dir="/home/studinstru/dlGenAI/dlgenai-t22026/models/roberta-base",
    allow_patterns=common_patterns,
)

print("DeBERTa (safetensors) at:", deberta_path)
print("RoBERTa (safetensors) at:", roberta_path)

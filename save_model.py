from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     local_dir="models/Qwen2.5-7B-Instruct",
#     local_dir_use_symlinks=False,
#     ignore_patterns=["original/*"],
# )


snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    local_dir="models/Qwen2.5-3B-Instruct",
    local_dir_use_symlinks=False,
    ignore_patterns=["original/*"],
)
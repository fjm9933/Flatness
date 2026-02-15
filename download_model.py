import os
from huggingface_hub import snapshot_download


# Your HF token (requires access to LLaMA)
hf_token = ""

# Target directory
target_dir = "/your_path/pretrained_models/llama-3-8b-instruct"

# Download the model
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    local_dir=target_dir,
    local_dir_use_symlinks=False,     # Do not use symlinks
    resume_download=True,             # Resume download if interrupted
    token=hf_token,
    allow_patterns=[                  # Only download files with these patterns
        "*.model",
        "*.json",
        "*.safetensors",
        "*.py",
        "*.txt",
        "*.md"
    ],
    ignore_patterns=[                 # Exclude these file patterns
        "*.msgpack",
        "*.bin",
        "*.h5",
        "*.ot"
    ]
)

print(f"✅ Model downloaded to {target_dir}")
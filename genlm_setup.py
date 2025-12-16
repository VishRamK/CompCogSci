"""
GenLM Control + Llama 3.2-Instruct setup script.
- Installs required packages if missing
- Handles HuggingFace login
- Loads Llama 3.2-Instruct for use in downstream scripts
"""
import os
import subprocess
import sys

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
for pkg in [
    "genlm-control",
    "transformers>=4.44.0",
    "torch",
    "huggingface_hub",
    "jsonschema",
    "lark",
    "arsenal",
    "python-dotenv"
    "asyncio"
    "ludax"
]:
    try:
        __import__(pkg.split('>=')[0].replace('-', '_'))
    except ImportError:
        pip_install(pkg)

# HuggingFace login
from huggingface_hub import login
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=True)
    print("✅ HuggingFace login successful.")
else:
    print("⚠️  Set HF_TOKEN env var or run login() manually if needed.")

print("GenLM Control and dependencies are ready.")

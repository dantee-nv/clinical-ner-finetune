# Colab Quickstart (Single Cell)

Use this in **one Colab cell** after selecting `Runtime -> Change runtime type -> T4 GPU`.

```python
# ===== clinical-ner-finetune: one-cell Colab run =====
REPO_URL = "https://github.com/dantee-nv/clinical-ner-finetune.git"
GITHUB_TOKEN = ""  # leave blank for public repos; set for private repos
MAX_SAMPLES = 1500  # lower to 500 for faster testing

import os
import shutil
import subprocess

clone_url = REPO_URL
if GITHUB_TOKEN:
    clone_url = REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@")

if os.path.exists("/content/clinical-ner-finetune"):
    shutil.rmtree("/content/clinical-ner-finetune")

subprocess.run(["git", "clone", clone_url, "/content/clinical-ner-finetune"], check=True)
os.chdir("/content/clinical-ner-finetune")
assert os.path.exists("data/parse_ccda.py"), "Clone failed: data/parse_ccda.py not found."
print("Working directory:", os.getcwd())

!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt

# 1) XML parsing demo
!python data/parse_ccda.py \
  --input_xml data/raw/sample_ccda.xml \
  --output_txt data/interim/parsed_ccda.txt \
  --output_json data/interim/parsed_ccda_sections.json

# 2) Public dataset curation (with fallback order)
!python data/curate.py \
  --output_dir data/processed \
  --dataset_preference medical_ner,ncbi_disease,bc5cdr \
  --seed 42 \
  --max_samples {MAX_SAMPLES}

# 3) Data quality validation
!python data/validate.py --input_glob "data/processed/*.jsonl"

# 4) QLoRA fine-tuning
!python train/train.py \
  --config train/config.yaml \
  --output_dir outputs/phi3-clinical-ner

# 5) Held-out evaluation (adapter predictor)
!python eval/evaluate.py \
  --test_file data/processed/test.jsonl \
  --base_model microsoft/Phi-3-mini-4k-instruct \
  --adapter_dir outputs/phi3-clinical-ner/adapter \
  --predictor adapter \
  --output_file eval/results.json

# 6) Inference demo
!python inference/predict.py \
  --base_model microsoft/Phi-3-mini-4k-instruct \
  --adapter_dir outputs/phi3-clinical-ner/adapter \
  --note "Patient with COPD started tiotropium and had pulmonary function testing."

# 7) Bundle artifacts for download
!zip -r clinical-ner-artifacts.zip \
  outputs/phi3-clinical-ner \
  eval/results.json \
  data/processed/train.jsonl \
  data/processed/val.jsonl \
  data/processed/test.jsonl \
  data/interim/parsed_ccda.txt \
  data/interim/parsed_ccda_sections.json

from google.colab import files
files.download("clinical-ner-artifacts.zip")
```

### Private Repo Token (Recommended)
If your repo is private, create a GitHub personal access token with `repo` access and set `GITHUB_TOKEN` in the cell above.

## If You Don’t Use GitHub
- Upload your local `clinical-ner-finetune` folder to Colab manually (or via Drive).
- Then run the same commands starting from `%cd clinical-ner-finetune`.

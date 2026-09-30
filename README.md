# Acadify Fine-Tuning Pipeline

A production-oriented supervised fine-tuning (SFT) pipeline for causal language models with LoRA/QLoRA, Accelerate, DeepSpeed ZeRO-3, dataset normalization, run manifests, and AWS SageMaker/Terraform integration.

Core training stays framework-native: Hugging Face Transformers + PEFT + Accelerate.

## Capabilities

| Area | Capability |
|---|---|
| Fine-tuning | LoRA / QLoRA, assistant-only loss masking |
| Models | Hugging Face causal LMs with explicit chat-template validation |
| Scaling | Single GPU, multi-GPU, Accelerate, optional DeepSpeed |
| Data | JSON/JSONL/HF normalization, schema filtering, deterministic split |
| Reproducibility | Seed control and per-run run_manifest.json |
| Artifacts | Safe-serialization model checkpoints |
| Quality | Ruff, Black, config validation, tests, CodeQL, Terraform validation |

## Quick start

```bash
git clone https://github.com/AcadifySolution/fine-tuning-pipeline.git
cd fine-tuning-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Repository checks:
```bash
pip install -r requirements-dev.txt
python -m compileall -q data src
ruff check data src
black --check data src
pytest -q
```

## Data preparation

```bash
python data/preprocess.py \
  --dataset_name "timdettmers/openassistant-guanaco" \
  --output_dir data/processed \
  --test_split_size 0.1 \
  --seed 42
```

Output is normalized to a messages schema. Production datasets should be reviewed for provenance, licensing, duplicates, sensitive information, and task-specific quality.

## Training

```bash
python src/train.py \
  --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
  --train_path data/processed/train.jsonl \
  --val_path data/processed/val.jsonl \
  --lora_config configs/lora_config.yaml \
  --output_dir checkpoints/sft_model \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --max_seq_length 2048
```

Each run writes run_manifest.json with the model, dataset paths, seed, sequence length, dataset counts, and runtime metadata.

### Distributed / DeepSpeed
```bash
accelerate config
accelerate launch src/train.py \
  --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
  --train_path data/processed/train.jsonl \
  --val_path data/processed/val.jsonl \
  --lora_config configs/lora_config.yaml \
  --deepspeed_config configs/deepspeed_zero3.json \
  --output_dir checkpoints/deepspeed
```

### Merge LoRA
```bash
python src/merge_peft.py \
  --base_model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
  --adapter_dir checkpoints/sft_model \
  --output_dir checkpoints/merged_model
```

## AWS / SageMaker

Terraform is deployment scaffolding, not a universal production template. Review IAM scope, networking, model/container compatibility, GPU and regional availability, S3 policy scope, endpoint cost, and model licensing before apply.

```bash
cd terraform
terraform init -backend=false
terraform fmt -check -recursive
terraform validate
terraform plan
```

## Security

Never commit credentials, datasets, checkpoints, or Terraform state. Keep trust_remote_code disabled unless the required source code has been reviewed. Pin or lock dependencies in production environments.

See [SECURITY.md](SECURITY.md).

## License
MIT — see [LICENSE](LICENSE).
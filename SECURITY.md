# Security

This repository handles external model/dataset sources and generates high-value model artifacts.

## Baseline controls

- Never commit Hugging Face, AWS, or other credentials.
- Review model and dataset provenance, license, and remote-code requirements.
- Keep `trust_remote_code=False` unless required code has been reviewed.
- Keep datasets and checkpoints outside Git.
- Use short-lived AWS credentials and least-privilege IAM roles.
- Validate Terraform before deployment and review all IAM/network changes.
- Treat training data, logs, and checkpoints as potentially sensitive.
- Pin or lock production dependencies and record the environment used for each training run.
- Validate task-specific quality and safety before deploying a fine-tuned model.

## Reporting

Report suspected security issues privately to the maintainers rather than publishing exploit details in an issue.

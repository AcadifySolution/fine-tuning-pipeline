# Security

## Scope

This repository trains and packages models from external model and dataset sources. Security therefore includes source trust, dependency integrity, dataset handling, credentials, artifact integrity, and cloud IAM.

## Rules

- Never commit Hugging Face, AWS, or other access tokens.
- Treat model and dataset identifiers as untrusted inputs and review licenses/terms before use.
- Keep `trust_remote_code=False` unless a specific model requires reviewed custom code.
- Keep datasets and checkpoints outside Git; the repository ignores common model/data artifacts.
- Prefer short-lived AWS credentials and least-privilege IAM roles.
- Do not expose generated checkpoints or logs publicly unless intentionally published.
- Pin or lock production dependencies in a deployment environment before repeatable production training.
- Validate Terraform before applying it and review IAM/network changes.
- Do not treat validation loss alone as model quality; pair it with task-specific evaluation before deployment.

## Reporting

Report suspected vulnerabilities privately to the repository maintainers rather than publishing exploit details in an issue.

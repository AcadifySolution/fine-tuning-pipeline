variable "aws_region" {
  type        = string
  description = "The target AWS Region for all training and deployment resources."
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Name prefix for resources associated with this project."
  default     = "acadify-llm-finetuning"
}

variable "environment" {
  type        = string
  description = "Deployment environment lifecycle tag."
  default     = "production"
}

variable "sagemaker_instance_type" {
  type        = string
  description = "The EC2 instance type utilized for running the SageMaker Training Job."
  default     = "ml.p4d.24xlarge" # 8x NVIDIA A100 GPUs (ideal for FSDP / DeepSpeed ZeRO-3)
}

variable "sagemaker_inference_instance_type" {
  type        = string
  description = "The EC2 instance type utilized for SageMaker real-time inference endpoints."
  default     = "ml.g5.2xlarge" # 1x NVIDIA A10G GPU (ideal for inference of LoRA adapted Llama 3)
}

variable "model_id" {
  type        = string
  description = "The base model identifier for the model hub."
  default     = "meta-llama/Meta-Llama-3-8B-Instruct"
}

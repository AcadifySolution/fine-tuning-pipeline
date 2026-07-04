# Terraform configuration for AWS SageMaker LLM Fine-Tuning Pipeline
# Author: Acadify Solution

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ==========================================
# S3 BUCKET FOR DATASETS & MODEL ARTIFACTS
# ==========================================

resource "aws_s3_bucket" "model_bucket" {
  bucket        = "${var.project_name}-bucket-${var.environment}"
  force_destroy = false

  tags = {
    Name        = "${var.project_name}-s3-bucket"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_s3_bucket_public_access_block" "block_public" {
  bucket = aws_s3_bucket.model_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ==========================================
# IAM ROLES AND POLICIES FOR SAGEMAKER
# ==========================================

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.project_name}-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# IAM Policy for SageMaker to read/write to S3 and CloudWatch Logs
resource "aws_iam_policy" "sagemaker_policy" {
  name        = "${var.project_name}-policy"
  description = "Execution policy for LLM fine-tuning and inference endpoints on SageMaker."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.model_bucket.arn,
          "${aws_s3_bucket.model_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_attach" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_policy.arn
}

# Attach AmazonSageMakerFullAccess for administrative and container provisioning compatibility
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# ==========================================
# SAGEMAKER NOTEBOOK INSTANCE (For experiment orchestration)
# ==========================================

resource "aws_sagemaker_notebook_instance" "notebook" {
  name          = "${var.project_name}-dev-notebook"
  role_arn      = aws_iam_role.sagemaker_execution_role.arn
  instance_type = "ml.t3.medium"

  tags = {
    Name        = "${var.project_name}-notebook"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# ==========================================
# SAGEMAKER MODEL REGISTRY & ENDPOINT DEPLOYMENT
# ==========================================

# SageMaker Model definition linking to the S3 adapter artifacts
resource "aws_sagemaker_model" "finetuned_model" {
  name               = "${var.project_name}-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn

  primary_container {
    image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04" # Official HF TGI Container
    model_data_url = "s3://${aws_s3_bucket.model_bucket.id}/checkpoints/sft_model/model.tar.gz"
    
    environment = {
      HF_MODEL_ID      = var.model_id
      HF_TASK          = "text-generation"
      NUMBER_OF_GPUS   = "1"
      MAX_INPUT_LENGTH = "2048"
      MAX_TOTAL_TOKENS = "4096"
    }
  }

  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "endpoint_config" {
  name = "${var.project_name}-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.finetuned_model.name
    initial_instance_count = 1
    instance_type          = var.sagemaker_inference_instance_type
    initial_variant_weight = 1.0
  }

  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Real-Time Endpoint deployment
resource "aws_sagemaker_endpoint" "inference_endpoint" {
  name                 = "${var.project_name}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.endpoint_config.name

  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.model_bucket.id
  description = "The name of the S3 bucket created for dataset storage and checkpoint output."
}

output "sagemaker_execution_role_arn" {
  value       = aws_iam_role.sagemaker_execution_role.arn
  description = "The ARN of the IAM role utilized by SageMaker for training and deployment."
}

output "sagemaker_notebook_url" {
  value       = "https://${aws_sagemaker_notebook_instance.notebook.name}.notebook.${var.aws_region}.sagemaker.aws"
  description = "The direct URL access pathway to open the SageMaker Jupyter notebook environment."
}

output "sagemaker_endpoint_name" {
  value       = aws_sagemaker_endpoint.inference_endpoint.name
  description = "The name of the deployed SageMaker real-time inference endpoint."
}

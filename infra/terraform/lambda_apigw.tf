# ============================================================
# Terraform — AWS Lambda + API Gateway for Document Portal
#
# Creates:
#   - Lambda function (health + analyze endpoints)
#   - API Gateway REST API in front of Lambda
#   - IAM role for Lambda execution
#
# Usage:
#   cd infra/terraform
#   terraform apply -target=aws_lambda_function.portal
#
# After apply, package and deploy:
#   cd infra/lambda
#   pip install -r ../../requirements.txt -t package/
#   cp handler.py package/
#   cd package && zip -r ../lambda_package.zip . && cd ..
#   aws lambda update-function-code \
#     --function-name document-portal-lambda \
#     --zip-file fileb://lambda_package.zip
# ============================================================

# ── IAM Role for Lambda ───────────────────────────────────────

resource "aws_iam_role" "lambda_exec" {
  name = "document-portal-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ── Lambda Function ───────────────────────────────────────────

resource "aws_lambda_function" "portal" {
  function_name = "document-portal-lambda"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.10"
  timeout       = 60
  memory_size   = 512

  # Placeholder — deploy real package via CLI after first apply
  filename      = "${path.module}/../lambda/lambda_package.zip"

  environment {
    variables = {
      ENV          = "production"
      LLM_PROVIDER = "openai"
    }
  }
}

# ── API Gateway ───────────────────────────────────────────────

resource "aws_api_gateway_rest_api" "portal" {
  name        = "document-portal-api"
  description = "API Gateway for Document Portal Lambda"
}

resource "aws_api_gateway_resource" "health" {
  rest_api_id = aws_api_gateway_rest_api.portal.id
  parent_id   = aws_api_gateway_rest_api.portal.root_resource_id
  path_part   = "health"
}

resource "aws_api_gateway_resource" "analyze" {
  rest_api_id = aws_api_gateway_rest_api.portal.id
  parent_id   = aws_api_gateway_rest_api.portal.root_resource_id
  path_part   = "analyze"
}

resource "aws_api_gateway_method" "health_get" {
  rest_api_id   = aws_api_gateway_rest_api.portal.id
  resource_id   = aws_api_gateway_resource.health.id
  http_method   = "GET"
  authorization = "NONE"
}

resource "aws_api_gateway_method" "analyze_post" {
  rest_api_id   = aws_api_gateway_rest_api.portal.id
  resource_id   = aws_api_gateway_resource.analyze.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "health" {
  rest_api_id             = aws_api_gateway_rest_api.portal.id
  resource_id             = aws_api_gateway_resource.health.id
  http_method             = aws_api_gateway_method.health_get.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.portal.invoke_arn
}

resource "aws_api_gateway_integration" "analyze" {
  rest_api_id             = aws_api_gateway_rest_api.portal.id
  resource_id             = aws_api_gateway_resource.analyze.id
  http_method             = aws_api_gateway_method.analyze_post.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.portal.invoke_arn
}

resource "aws_api_gateway_deployment" "portal" {
  rest_api_id = aws_api_gateway_rest_api.portal.id
  depends_on  = [
    aws_api_gateway_integration.health,
    aws_api_gateway_integration.analyze,
  ]
}

resource "aws_api_gateway_stage" "prod" {
  rest_api_id   = aws_api_gateway_rest_api.portal.id
  deployment_id = aws_api_gateway_deployment.portal.id
  stage_name    = "prod"
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.portal.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.portal.execution_arn}/*/*"
}

output "api_gateway_url" {
  value = "${aws_api_gateway_stage.prod.invoke_url}"
}

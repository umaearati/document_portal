# ============================================================
# Terraform — Document Portal on AWS ECS Fargate
# Replicates the existing GitHub Actions / ECS deployment
# as infrastructure-as-code.
#
# Usage:
#   cd infra/terraform
#   terraform init
#   terraform plan
#   terraform apply
#
# Required env vars (set before running):
#   export TF_VAR_openai_api_key="sk-..."
#   export TF_VAR_google_api_key="AIza..."
#   export TF_VAR_groq_api_key="gsk_..."
#   export TF_VAR_langfuse_public_key="pk-lf-..."
#   export TF_VAR_langfuse_secret_key="sk-lf-..."
# ============================================================

terraform {
  required_version = ">= 1.5"
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

# ── Variables ────────────────────────────────────────────────

variable "aws_region"           { default = "eu-west-2" }
variable "app_name"             { default = "document-portal" }
variable "container_port"       { default = 8080 }
variable "cpu"                  { default = "512" }
variable "memory"               { default = "1024" }
variable "openai_api_key"       { sensitive = true }
variable "google_api_key"       { sensitive = true }
variable "groq_api_key"         { sensitive = true }
variable "langfuse_public_key"  { sensitive = true; default = "" }
variable "langfuse_secret_key"  { sensitive = true; default = "" }
variable "db_password"          { sensitive = true; default = "changeme123" }

# ── Data sources ─────────────────────────────────────────────

data "aws_caller_identity" "current" {}
data "aws_vpc" "default" { default = true }
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ── ECR Repository ───────────────────────────────────────────

resource "aws_ecr_repository" "app" {
  name                 = var.app_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ── ECS Cluster ──────────────────────────────────────────────

resource "aws_ecs_cluster" "main" {
  name = "${var.app_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ── IAM Role for ECS Task ────────────────────────────────────

resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.app_name}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── Secrets Manager ──────────────────────────────────────────

resource "aws_secretsmanager_secret" "api_keys" {
  name = "${var.app_name}/api-keys"
}

resource "aws_secretsmanager_secret_version" "api_keys" {
  secret_id = aws_secretsmanager_secret.api_keys.id
  secret_string = jsonencode({
    OPENAI_API_KEY      = var.openai_api_key
    GOOGLE_API_KEY      = var.google_api_key
    GROQ_API_KEY        = var.groq_api_key
    LANGFUSE_PUBLIC_KEY = var.langfuse_public_key
    LANGFUSE_SECRET_KEY = var.langfuse_secret_key
  })
}

# ── RDS PostgreSQL ───────────────────────────────────────────

resource "aws_db_instance" "postgres" {
  identifier           = "${var.app_name}-db"
  engine               = "postgres"
  engine_version       = "16"
  instance_class       = "db.t3.micro"    # free tier eligible
  allocated_storage    = 20
  db_name              = "document_portal"
  username             = "postgres"
  password             = var.db_password
  skip_final_snapshot  = true
  publicly_accessible  = false
  deletion_protection  = false

  tags = { Name = "${var.app_name}-postgres" }
}

# ── Security Group ───────────────────────────────────────────

resource "aws_security_group" "app" {
  name   = "${var.app_name}-sg"
  vpc_id = data.aws_vpc.default.id

  ingress {
    from_port   = var.container_port
    to_port     = var.container_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── CloudWatch Log Group ─────────────────────────────────────

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 30
}

# ── ECS Task Definition ──────────────────────────────────────

resource "aws_ecs_task_definition" "app" {
  family                   = var.app_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([{
    name  = "${var.app_name}-container"
    image = "${aws_ecr_repository.app.repository_url}:latest"

    portMappings = [{
      containerPort = var.container_port
      protocol      = "tcp"
    }]

    environment = [
      { name = "ENV",          value = "production" },
      { name = "LLM_PROVIDER", value = "openai" },
      { name = "DATABASE_URL", value = "postgresql://postgres:${var.db_password}@${aws_db_instance.postgres.address}:5432/document_portal" },
      { name = "PII_ENABLED",  value = "true" },
    ]

    secrets = [{
      name      = "API_KEYS"
      valueFrom = aws_secretsmanager_secret.api_keys.arn
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "ecs"
      }
    }
  }])
}

# ── ECS Service ──────────────────────────────────────────────

resource "aws_ecs_service" "app" {
  name            = "${var.app_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true
  }
}

# ── Outputs ──────────────────────────────────────────────────

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "db_endpoint" {
  value = aws_db_instance.postgres.address
}

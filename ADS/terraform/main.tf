terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "s3" {
    bucket = "ads-terraform-state"
    key    = "ads/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC
resource "aws_vpc" "ads" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "ads-vpc"
  }
}

# Subnets
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.ads.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "ads-public-${count.index}"
  }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.ads.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "ads-private-${count.index}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "ads" {
  vpc_id = aws_vpc.ads.id

  tags = {
    Name = "ads-igw"
  }
}

# NAT Gateway
resource "aws_eip" "nat" {
  domain = "vpc"
}

resource "aws_nat_gateway" "ads" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name = "ads-nat"
  }
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.ads.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.ads.id
  }

  tags = {
    Name = "ads-public-rt"
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.ads.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.ads.id
  }

  tags = {
    Name = "ads-private-rt"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "ads-cluster"
  cluster_version = "1.31"

  vpc_id     = aws_vpc.ads.id
  subnet_ids = aws_subnet.private[*].id

  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    cpu_nodes = {
      desired_size = 2
      min_size     = 1
      max_size     = 5

      instance_types = ["c6i.4xlarge", "c6a.4xlarge", "m6i.4xlarge"]

      capacity_type = "ON_DEMAND"

      tags = {
        "k8s.io/cluster-autoscaler/node-template/label/node-type" = "cpu"
      }
    }
  }

  tags = {
    Environment = "production"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "ads" {
  identifier        = "ads-postgres"
  engine            = "postgres"
  engine_version    = "16.3"
  instance_class    = "db.r6i.large"
  allocated_storage = 100

  db_name  = "ads"
  username = "ads"
  password = random_password.db.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.ads.name

  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  deletion_protection = true

  tags = {
    Name = "ads-postgres"
  }
}

resource "random_password" "db" {
  length  = 24
  special = false
}

resource "aws_db_subnet_group" "ads" {
  name       = "ads-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "ads" {
  cluster_id           = "ads-redis"
  engine               = "redis"
  node_type            = "cache.r6g.large"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379

  subnet_group_name = aws_elasticache_subnet_group.ads.name
  security_group_ids = [aws_security_group.redis.id]

  tags = {
    Name = "ads-redis"
  }
}

resource "aws_elasticache_subnet_group" "ads" {
  name       = "ads-redis-subnet-group"
  subnet_ids = aws_subnet.private[*].id
}

# S3 Buckets
resource "aws_s3_bucket" "ads-models" {
  bucket = "ads-models-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "ads-data" {
  bucket = "ads-data-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "ads-logs" {
  bucket = "ads-logs-${data.aws_caller_identity.current.account_id}"
}

# Security Groups
resource "aws_security_group" "rds" {
  name   = "ads-rds-sg"
  vpc_id = aws_vpc.ads.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
}

resource "aws_security_group" "redis" {
  name   = "ads-redis-sg"
  vpc_id = aws_vpc.ads.id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
}

# ECR Repository
resource "aws_ecr_repository" "ads-api" {
  name                 = "ads-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# IAM roles for EKS
resource "aws_iam_role" "ads-s3-access" {
  name = "ads-s3-access-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = module.eks.oidc_provider_arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${module.eks.oidc_provider}:sub" = "system:serviceaccount:ads:ads-sa"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "ads-s3-access" {
  name = "ads-s3-access-policy"
  role = aws_iam_role.ads-s3-access.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.ads-models.arn,
          "${aws_s3_bucket.ads-models.arn}/*",
          aws_s3_bucket.ads-data.arn,
          "${aws_s3_bucket.ads-data.arn}/*",
        ]
      }
    ]
  })
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# Outputs
output "eks_cluster_name" {
  value = module.eks.cluster_name
}

output "rds_endpoint" {
  value = aws_db_instance.ads.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.ads.cache_nodes[0].address
}

output "ecr_repository_url" {
  value = aws_ecr_repository.ads-api.repository_url
}

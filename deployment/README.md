# Deployment Directory

## Overview
The `deployment/` directory contains deployment configurations, scripts, and documentation for deploying DataScribe in various environments. It covers local development, staging, and production deployments with different deployment strategies and infrastructure options.

## 📁 Contents

### `docker/`
**Docker Containerization**

Docker-based deployment configurations:
- **`Dockerfile`**: Main application container definition
- **`docker-compose.yml`**: Multi-service development environment
- **`docker-compose.prod.yml`**: Production-ready multi-service setup
- **`.dockerignore`**: Files to exclude from Docker build
- **`nginx/`**: Nginx reverse proxy configuration
- **`postgres/`**: PostgreSQL database configuration

### `kubernetes/`
**Kubernetes Deployment**

Kubernetes manifests and configurations:
- **`deployment.yaml`**: Main application deployment
- **`service.yaml`**: Service definitions
- **`ingress.yaml`**: Ingress configuration
- **`configmap.yaml`**: Configuration management
- **`secrets.yaml`**: Secret management
- **`namespace.yaml`**: Namespace definition
- **`rbac.yaml`**: Role-based access control

### `terraform/`
**Infrastructure as Code**

Terraform configurations for cloud infrastructure:
- **`main.tf`**: Main infrastructure configuration
- **`variables.tf`**: Variable definitions
- **`outputs.tf`**: Output values
- **`providers.tf`**: Provider configurations
- **`modules/`**: Reusable infrastructure modules
- **`environments/`**: Environment-specific configurations

### `ansible/`
**Configuration Management**

Ansible playbooks for server configuration:
- **`playbook.yml`**: Main server setup playbook
- **`inventory/`**: Server inventory files
- **`roles/`**: Reusable Ansible roles
- **`vars/`**: Variable definitions
- **`templates/`**: Configuration templates

### `scripts/`
**Deployment Scripts**

Automated deployment and management scripts:
- **`deploy.sh`**: Main deployment script
- **`setup.sh`**: Initial server setup
- **`backup.sh`**: Database and file backup
- **`monitoring.sh`**: Monitoring setup
- **`ssl.sh`**: SSL certificate management
- **`update.sh`**: Application update script

### `configs/`
**Deployment Configurations**

Environment-specific configuration files:
- **`production.env`**: Production environment variables
- **`staging.env`**: Staging environment variables
- **`development.env`**: Development environment variables
- **`nginx.conf`**: Nginx server configuration
- **`gunicorn.conf.py`**: Gunicorn WSGI configuration
- **`supervisor.conf`**: Process management configuration

## 🐳 Docker Deployment

### Dockerfile
**Application Container Definition**

```dockerfile
# deployment/docker/Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash datascribe \
    && chown -R datascribe:datascribe /app
USER datascribe

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
```

### Docker Compose
**Multi-Service Development Environment**

```yaml
# deployment/docker/docker-compose.yml
version: '3.8'

services:
  datascribe:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATASCRIBE_ENV=development
      - DATASCRIBE_DEBUG=true
      - DATABASE_URL=postgresql://datascribe:password@db:5432/datascribe
    volumes:
      - ./uploads:/app/uploads
      - ./reports:/app/reports
    depends_on:
      - db
      - redis
    networks:
      - datascribe-network

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=datascribe
      - POSTGRES_USER=datascribe
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - datascribe-network

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    networks:
      - datascribe-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - datascribe
    networks:
      - datascribe-network

volumes:
  postgres_data:

networks:
  datascribe-network:
    driver: bridge
```

## ☸️ Kubernetes Deployment

### Main Deployment
**Application Deployment Manifest**

```yaml
# deployment/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: datascribe
  namespace: datascribe
  labels:
    app: datascribe
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: datascribe
  template:
    metadata:
      labels:
        app: datascribe
        version: v1.0.0
    spec:
      containers:
      - name: datascribe
        image: datascribe:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATASCRIBE_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: datascribe-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: datascribe-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: uploads
          mountPath: /app/uploads
        - name: reports
          mountPath: /app/reports
      volumes:
      - name: uploads
        persistentVolumeClaim:
          claimName: datascribe-uploads-pvc
      - name: reports
        persistentVolumeClaim:
          claimName: datascribe-reports-pvc
```

### Service Definition
**Kubernetes Service Configuration**

```yaml
# deployment/kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: datascribe-service
  namespace: datascribe
spec:
  selector:
    app: datascribe
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: datascribe-db
  namespace: datascribe
spec:
  selector:
    app: postgres
  ports:
  - protocol: TCP
    port: 5432
    targetPort: 5432
  type: ClusterIP
```

### Ingress Configuration
**External Access Configuration**

```yaml
# deployment/kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: datascribe-ingress
  namespace: datascribe
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - datascribe.yourdomain.com
    secretName: datascribe-tls
  rules:
  - host: datascribe.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: datascribe-service
            port:
              number: 80
```

## 🏗️ Terraform Infrastructure

### Main Configuration
**Infrastructure as Code**

```hcl
# deployment/terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc"
  
  vpc_name = "${var.project_name}-vpc"
  vpc_cidr = var.vpc_cidr
  azs      = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"
  
  cluster_name    = "${var.project_name}-cluster"
  cluster_version = var.kubernetes_version
  vpc_id         = module.vpc.vpc_id
  subnet_ids     = module.vpc.private_subnet_ids
  
  node_groups = {
    general = {
      desired_capacity = 2
      max_capacity     = 4
      min_capacity     = 1
      
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
    }
  }
}

# RDS Database
module "rds" {
  source = "./modules/rds"
  
  identifier = "${var.project_name}-db"
  engine     = "postgres"
  version    = "13.7"
  
  instance_class = "db.t3.micro"
  allocated_storage = 20
  
  db_name  = "datascribe"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [module.vpc.default_security_group_id]
  subnet_ids             = module.vpc.private_subnet_ids
}

# S3 Bucket for Storage
resource "aws_s3_bucket" "datascribe_storage" {
  bucket = "${var.project_name}-storage-${random_string.bucket_suffix.result}"
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}
```

### Variables
**Terraform Variable Definitions**

```hcl
# deployment/terraform/variables.tf
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "datascribe"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.24"
}

variable "db_username" {
  description = "Database username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}
```

## 🔧 Ansible Configuration

### Main Playbook
**Server Configuration Management**

```yaml
# deployment/ansible/playbook.yml
---
- name: Configure DataScribe Server
  hosts: datascribe_servers
  become: yes
  vars:
    datascribe_user: datascribe
    datascribe_group: datascribe
    app_directory: /opt/datascribe
    python_version: "3.9"
  
  tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
        cache_valid_time: 3600
    
    - name: Install system packages
      apt:
        name:
          - python{{ python_version }}
          - python{{ python_version }}-pip
          - python{{ python_version }}-venv
          - postgresql-client
          - nginx
          - supervisor
          - curl
          - git
        state: present
    
    - name: Create datascribe user
      user:
        name: "{{ datascribe_user }}"
        group: "{{ datascribe_group }}"
        system: yes
        shell: /bin/bash
        home: /home/{{ datascribe_user }}
    
    - name: Create application directory
      file:
        path: "{{ app_directory }}"
        state: directory
        owner: "{{ datascribe_user }}"
        group: "{{ datascribe_group }}"
        mode: '0755'
    
    - name: Setup Python virtual environment
      pip:
        virtualenv: "{{ app_directory }}/venv"
        virtualenv_command: python{{ python_version }} -m venv
        requirements: "{{ app_directory }}/requirements.txt"
        state: present
    
    - name: Configure Nginx
      template:
        src: templates/nginx.conf.j2
        dest: /etc/nginx/sites-available/datascribe
        owner: root
        group: root
        mode: '0644'
      notify: restart nginx
    
    - name: Enable Nginx site
      file:
        src: /etc/nginx/sites-available/datascribe
        dest: /etc/nginx/sites-enabled/datascribe
        state: link
      notify: restart nginx
    
    - name: Configure Supervisor
      template:
        src: templates/supervisor.conf.j2
        dest: /etc/supervisor/conf.d/datascribe.conf
        owner: root
        group: root
        mode: '0644'
      notify: restart supervisor
    
    - name: Start and enable services
      systemd:
        name: "{{ item }}"
        state: started
        enabled: yes
      loop:
        - nginx
        - supervisor
  
  handlers:
    - name: restart nginx
      service:
        name: nginx
        state: restarted
    
    - name: restart supervisor
      service:
        name: supervisor
        state: restarted
```

## 🚀 Deployment Scripts

### Main Deployment Script
**Automated Deployment Process**

```bash
#!/bin/bash
# deployment/scripts/deploy.sh

set -e

# Configuration
APP_NAME="datascribe"
DEPLOYMENT_ENV="${1:-production}"
DOCKER_REGISTRY="your-registry.com"
IMAGE_TAG="${2:-latest}"

echo "🚀 Deploying $APP_NAME to $DEPLOYMENT_ENV environment..."

# Load environment variables
source "configs/${DEPLOYMENT_ENV}.env"

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t $DOCKER_REGISTRY/$APP_NAME:$IMAGE_TAG .
docker push $DOCKER_REGISTRY/$APP_NAME:$IMAGE_TAG

# Deploy to Kubernetes
if [ "$DEPLOYMENT_ENV" = "production" ] || [ "$DEPLOYMENT_ENV" = "staging" ]; then
    echo "☸️  Deploying to Kubernetes..."
    
    # Update image tag in deployment
    sed -i "s|image: .*|image: $DOCKER_REGISTRY/$APP_NAME:$IMAGE_TAG|" \
        kubernetes/deployment.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f kubernetes/namespace.yaml
    kubectl apply -f kubernetes/configmap.yaml
    kubectl apply -f kubernetes/secrets.yaml
    kubectl apply -f kubernetes/deployment.yaml
    kubectl apply -f kubernetes/service.yaml
    kubectl apply -f kubernetes/ingress.yaml
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/$APP_NAME -n $APP_NAME
    
    echo "✅ Kubernetes deployment completed!"
    
else
    echo "🐳 Deploying with Docker Compose..."
    
    # Update environment file
    export IMAGE_TAG=$IMAGE_TAG
    
    # Deploy with docker-compose
    docker-compose -f docker/docker-compose.yml down
    docker-compose -f docker/docker-compose.yml up -d
    
    echo "✅ Docker Compose deployment completed!"
fi

# Health check
echo "🏥 Performing health check..."
sleep 10
curl -f http://localhost:8000/health || {
    echo "❌ Health check failed!"
    exit 1
}

echo "🎉 Deployment completed successfully!"
echo "🌐 Application is available at: $APP_URL"
```

### Backup Script
**Data Backup and Recovery**

```bash
#!/bin/bash
# deployment/scripts/backup.sh

set -e

# Configuration
BACKUP_DIR="/backups/datascribe"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

echo "💾 Starting backup process..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Database backup
echo "🗄️  Backing up database..."
pg_dump $DATABASE_URL > "$BACKUP_DIR/database_$DATE.sql"
gzip "$BACKUP_DIR/database_$DATE.sql"

# File backups
echo "📁 Backing up uploads and reports..."
tar -czf "$BACKUP_DIR/files_$DATE.tar.gz" \
    -C /opt/datascribe uploads reports

# Configuration backup
echo "⚙️  Backing up configuration..."
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" \
    -C /opt/datascribe config .env

# Cleanup old backups
echo "🧹 Cleaning up old backups..."
find "$BACKUP_DIR" -name "*.gz" -mtime +$RETENTION_DAYS -delete

echo "✅ Backup completed successfully!"
echo "📊 Backup size: $(du -sh $BACKUP_DIR | cut -f1)"
echo "🗂️  Backup location: $BACKUP_DIR"
```

## 📊 Monitoring and Logging

### Prometheus Configuration
**Metrics Collection**

```yaml
# deployment/configs/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "datascribe_rules.yml"

scrape_configs:
  - job_name: 'datascribe'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'nginx'
    static_configs:
      - targets: ['localhost:9113']
```

### Grafana Dashboard
**Monitoring Dashboard Configuration**

```json
{
  "dashboard": {
    "title": "DataScribe Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active Jobs",
        "type": "stat",
        "targets": [
          {
            "expr": "datascribe_active_jobs",
            "legendFormat": "Active Jobs"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Configuration

### SSL/TLS Setup
**Secure Communication**

```bash
#!/bin/bash
# deployment/scripts/ssl.sh

set -e

DOMAIN="datascribe.yourdomain.com"
EMAIL="admin@yourdomain.com"

echo "🔒 Setting up SSL/TLS for $DOMAIN..."

# Install certbot
apt-get update
apt-get install -y certbot python3-certbot-nginx

# Obtain SSL certificate
certbot --nginx \
    --non-interactive \
    --agree-tos \
    --email $EMAIL \
    --domains $DOMAIN

# Setup auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | crontab -

echo "✅ SSL/TLS setup completed!"
echo "🔐 Certificate will auto-renew daily at 12:00 PM"
```

## 🔮 Future Enhancements

### Advanced Deployment Features
- **Blue-Green Deployment**: Zero-downtime deployments
- **Canary Releases**: Gradual rollout strategies
- **Auto-scaling**: Dynamic resource allocation
- **Multi-region**: Geographic distribution
- **Disaster Recovery**: Backup and recovery automation

### Infrastructure Improvements
- **Service Mesh**: Istio or Linkerd integration
- **Observability**: Distributed tracing and logging
- **Security**: Advanced security scanning and compliance
- **Cost Optimization**: Resource usage optimization
- **Compliance**: SOC2, GDPR compliance automation

## 📚 Related Resources

- [Production Setup Guide](../docs/deployment/production-setup.md)
- [Docker Guide](../docs/deployment/docker.md)
- [Kubernetes Guide](../docs/deployment/kubernetes.md)
- [Monitoring Guide](../docs/deployment/monitoring.md)
- [Security Best Practices](../docs/deployment/security.md)

---

**Deployment Directory** - Production deployment configurations and automation for DataScribe

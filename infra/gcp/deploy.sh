#!/bin/bash
# ============================================================
# Deploy Document Portal to GCP Cloud Run
#
# Usage:
#   export GCP_PROJECT_ID=your-project-id
#   ./deploy.sh
# ============================================================

set -e

PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project)}
REGION="europe-west2"
SERVICE_NAME="document-portal"
REPO="document-portal"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/app"

echo "Deploying to project: $PROJECT_ID"

# Create Artifact Registry repo if it doesn't exist
gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="Document Portal images" \
  2>/dev/null || echo "Repo already exists"

# Authenticate Docker
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet

# Build and push
docker build -t $IMAGE:latest ../../
docker push $IMAGE:latest

# Store secrets in GCP Secret Manager
echo "Creating secrets (skip if already exist)..."
echo -n "$OPENAI_API_KEY"  | gcloud secrets create openai-api-key  --data-file=- 2>/dev/null || true
echo -n "$GOOGLE_API_KEY"  | gcloud secrets create google-api-key  --data-file=- 2>/dev/null || true
echo -n "$GROQ_API_KEY"    | gcloud secrets create groq-api-key    --data-file=- 2>/dev/null || true

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image=$IMAGE:latest \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=3 \
  --set-env-vars="ENV=production,LLM_PROVIDER=openai,PII_ENABLED=true" \
  --set-secrets="OPENAI_API_KEY=openai-api-key:latest,GOOGLE_API_KEY=google-api-key:latest,GROQ_API_KEY=groq-api-key:latest"

echo ""
echo "Deployed! Service URL:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"

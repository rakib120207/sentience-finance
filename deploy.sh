#!/bin/bash
# EquiVoice — Automated Cloud Run Deployment Script
# Run this once you have Google Cloud credentials set up

set -e

PROJECT_ID="your-google-cloud-project-id"
SERVICE_NAME="equivoice-backend"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Starting EquiVoice deployment..."

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t $IMAGE_NAME .

# Push to Google Container Registry
echo "☁️  Pushing to Google Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "🌐 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --set-env-vars GROQ_API_KEY=$GROQ_API_KEY

echo "✅ EquiVoice deployed successfully!"
echo "🔗 Getting service URL..."
gcloud run services describe $SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --format 'value(status.url)'
@echo off
echo Starting EquiVoice Cloud Run deployment...

set PROJECT_ID=your-google-cloud-project-id
set SERVICE_NAME=equivoice-backend
set REGION=us-central1
set IMAGE_NAME=gcr.io/%PROJECT_ID%/%SERVICE_NAME%

echo Building Docker image...
docker build -t %IMAGE_NAME% .

echo Pushing to Container Registry...
docker push %IMAGE_NAME%

echo Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% ^
  --image %IMAGE_NAME% ^
  --platform managed ^
  --region %REGION% ^
  --allow-unauthenticated ^
  --port 8080 ^
  --memory 512Mi

echo Deployment complete!
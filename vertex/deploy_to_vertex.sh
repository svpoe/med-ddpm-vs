#!/bin/bash

# Complete Vertex AI deployment script
# Usage: ./deploy_to_vertex.sh PROJECT_ID BUCKET_NAME

set -e  # Exit on any error

# Configuration
PROJECT_ID=${1:-"vs-segmentation"}
BUCKET_NAME=${2:-"vs-segmentation"}
REGION="us-central1"
IMAGE_NAME="vs-segmentation"
IMAGE_TAG="latest"

echo "🚀 Deploying VS fine-tuning to Vertex AI"
echo "📍 Project: $PROJECT_ID"
echo "📍 Bucket: $BUCKET_NAME"
echo "📍 Region: $REGION"

# Step 1: Set up gcloud
echo "⚙️  Setting up gcloud..."
gcloud config set project $PROJECT_ID
gcloud auth configure-docker

# Step 2: Create bucket and upload data
echo "📦 Setting up cloud storage..."
gsutil mb -l $REGION gs://$BUCKET_NAME 2>/dev/null || echo "Bucket exists"
gsutil -m cp -r dataset/vs gs://$BUCKET_NAME/data/
gsutil cp model/model_128.pt gs://$BUCKET_NAME/models/pretrained/

# Step 3: Build and push Docker image
echo "🐳 Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG .
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG

# Step 4: Submit test job
echo "🧪 Submitting test job..."
python submit_vertex_job.py \
    --project_id $PROJECT_ID \
    --bucket_name $BUCKET_NAME \
    --job_name vs-segmentation-test \
    --test_run

echo "✅ Test job submitted!"
echo ""
echo "📊 Monitor your job at:"
echo "   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
echo ""
echo "🔄 To submit full training job:"
echo "   python submit_vertex_job.py --project_id $PROJECT_ID --bucket_name $BUCKET_NAME"

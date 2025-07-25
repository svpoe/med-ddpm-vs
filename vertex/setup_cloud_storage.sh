#!/bin/bash

# Configuration - CHANGE THESE VALUES
PROJECT_ID="vs-segmentation"
BUCKET_NAME="vs-segmentation"
REGION="us-central1"

echo "🚀 Setting up Vertex AI training environment..."

# Set project
gcloud config set project $PROJECT_ID

# Create bucket if it doesn't exist
echo "📦 Creating storage bucket..."
gsutil mb -l $REGION gs://$BUCKET_NAME 2>/dev/null || echo "Bucket already exists"

# Upload VS dataset
echo "📤 Uploading VS dataset..."
gsutil -m cp -r dataset/vs gs://$BUCKET_NAME/data/

# Upload pretrained model
echo "📤 Uploading pretrained model..."
gsutil cp model/model_128.pt gs://$BUCKET_NAME/models/pretrained/

# Create directories for outputs
gsutil -m cp /dev/null gs://$BUCKET_NAME/outputs/.keep

echo "✅ Cloud storage setup complete!"
echo "📍 Bucket: gs://$BUCKET_NAME"
echo "📍 VS data: gs://$BUCKET_NAME/data/vs/"
echo "📍 Pretrained model: gs://$BUCKET_NAME/models/pretrained/model_128.pt"

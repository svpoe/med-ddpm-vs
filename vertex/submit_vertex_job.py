#!/usr/bin/env python3
"""
Submit VS fine-tuning job to Vertex AI
"""
from google.cloud import aiplatform
import argparse

def submit_training_job(
    project_id,
    region,
    bucket_name,
    job_name,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    epochs=50000,
    test_run=False
):
    """Submit training job to Vertex AI"""
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Adjust for test run
    if test_run:
        epochs = 100
        job_name = f"{job_name}-test"
        print("🧪 Submitting TEST job with 100 epochs")
    else:
        print(f"🚀 Submitting FULL training job with {epochs} epochs")
    
    # Container image URI (will be built and pushed)
    image_uri = f"gcr.io/{project_id}/vs-diffusion:latest"
    
    # Training arguments
    args = [
        "--bucket_name", bucket_name,
        "--epochs", str(epochs),
        "--with_condition",
        "--batchsize", "2" if not test_run else "1",
        "--save_and_sample_every", "1000" if not test_run else "50"
    ]
    
    # Create custom training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=image_uri,
        command=["python", "train_vs_cloud.py"],
        args=args,
        requirements=["google-cloud-storage>=2.0.0"],
        model_serving_container_image_uri=None,
    )
    
    # Submit job
    job.run(
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        replica_count=1,
        sync=False  # Don't wait for completion
    )
    
    print(f"✅ Job submitted: {job_name}")
    print(f"📊 Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
    return job

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True, help='Google Cloud Project ID')
    parser.add_argument('--region', default='us-central1', help='Region for training')
    parser.add_argument('--bucket_name', required=True, help='GCS bucket name')
    parser.add_argument('--job_name', default='vs-diffusion-finetune', help='Job name')
    parser.add_argument('--test_run', action='store_true', help='Run short test job')
    parser.add_argument('--machine_type', default='n1-standard-4', help='Machine type')
    parser.add_argument('--gpu_type', default='NVIDIA_TESLA_T4', help='GPU type')
    parser.add_argument('--epochs', type=int, default=50000, help='Training epochs')
    
    args = parser.parse_args()
    
    job = submit_training_job(
        project_id=args.project_id,
        region=args.region,
        bucket_name=args.bucket_name,
        job_name=args.job_name,
        machine_type=args.machine_type,
        accelerator_type=args.gpu_type,
        epochs=args.epochs,
        test_run=args.test_run
    )

if __name__ == "__main__":
    main()

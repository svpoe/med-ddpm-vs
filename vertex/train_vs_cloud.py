#!/usr/bin/env python3
"""
Cloud-optimized VS fine-tuning script for Vertex AI
"""
import os
import sys
import argparse
import torch

# Add the app directory to Python path for imports
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google.cloud import storage
from torchvision.transforms import Compose, Lambda

# Import Med-DDPM modules with proper path handling
try:
    from diffusion_model.trainer import GaussianDiffusion, Trainer
    from diffusion_model.unet import create_model
    from dataset import NiftiImageGenerator, NiftiPairImageGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path)
    # Try alternative import paths
    sys.path.append('/app')
    sys.path.append('.')
    from diffusion_model.trainer import GaussianDiffusion, Trainer
    from diffusion_model.unet import create_model
    from dataset import NiftiImageGenerator, NiftiPairImageGenerator

def download_from_gcs(bucket_name, source_path, dest_path):
    """Download files from Google Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    print(f"Downloading {source_path} to {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if source_path.endswith('/'):
        # Download directory
        blobs = bucket.list_blobs(prefix=source_path.rstrip('/'))
        downloaded_count = 0
        for blob in blobs:
            if not blob.name.endswith('/'):
                local_path = os.path.join(dest_path, blob.name[len(source_path):])
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob.download_to_filename(local_path)
                downloaded_count += 1
                if downloaded_count % 10 == 0:
                    print(f"  Downloaded {downloaded_count} files...")
        print(f"  Downloaded {downloaded_count} total files")
    else:
        # Download single file
        blob = bucket.blob(source_path)
        blob.download_to_filename(dest_path)
        print(f"  Downloaded {source_path}")

def upload_to_gcs(bucket_name, source_path, dest_path):
    """Upload files to Google Cloud Storage"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(dest_path)
        blob.upload_from_filename(source_path)
        print(f"Uploaded {source_path} to gs://{bucket_name}/{dest_path}")
    except Exception as e:
        print(f"Failed to upload {source_path}: {e}")

# Transform functions (same as your local version)
def tensor_float(t):
    return torch.tensor(t).float()

def normalize_range(t):
    return (t * 2) - 1

def add_batch_dim(t):
    return t.unsqueeze(0)

def transpose_depth(t):
    return t.transpose(3, 1)

def permute_dims(t):
    return t.permute(3, 0, 1, 2)

def main():
    parser = argparse.ArgumentParser()
    
    # Cloud-specific arguments
    parser.add_argument('--bucket_name', type=str, required=True, help='GCS bucket name')
    parser.add_argument('--data_path', type=str, default='data/vs/', help='Path to VS data in GCS')
    parser.add_argument('--model_path', type=str, default='models/pretrained/model_128.pt', help='Pretrained model path in GCS')
    parser.add_argument('--output_path', type=str, default='outputs/', help='Output path in GCS')
    
    # Training arguments (same as your local script)
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--depth_size', type=int, default=128)
    parser.add_argument('--num_channels', type=int, default=64)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--num_class_labels', type=int, default=3)
    parser.add_argument('--train_lr', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--timesteps', type=int, default=250)
    parser.add_argument('--save_and_sample_every', type=int, default=1000)
    parser.add_argument('--with_condition', action='store_true', default=True)
    
    args = parser.parse_args()
    
    print("🚀 Starting VS fine-tuning on Vertex AI")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create necessary directories
    os.makedirs('./dataset', exist_ok=True)
    os.makedirs('./model', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Download data and model from GCS
    print("📥 Downloading data from GCS...")
    download_from_gcs(args.bucket_name, args.data_path, './dataset/')
    download_from_gcs(args.bucket_name, args.model_path, './model/model_128.pt')
    
    # Verify downloads
    vs_mask_path = './dataset/vs/mask/'
    vs_image_path = './dataset/vs/image/'
    model_path = './model/model_128.pt'
    
    if not os.path.exists(vs_mask_path):
        raise FileNotFoundError(f"VS mask directory not found: {vs_mask_path}")
    if not os.path.exists(vs_image_path):
        raise FileNotFoundError(f"VS image directory not found: {vs_image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")
    
    print(f"✅ Found {len(os.listdir(vs_mask_path))} mask files")
    print(f"✅ Found {len(os.listdir(vs_image_path))} image files")
    print(f"✅ Found pretrained model: {os.path.getsize(model_path)} bytes")
    
    # Set up transforms
    transform = Compose([
        Lambda(tensor_float),
        Lambda(normalize_range),
        Lambda(add_batch_dim),
        Lambda(transpose_depth),
    ])
    
    input_transform = Compose([
        Lambda(tensor_float),
        Lambda(normalize_range),
        Lambda(permute_dims),
        Lambda(transpose_depth),
    ])
    
    # Create dataset
    print("📊 Creating dataset...")
    dataset = NiftiPairImageGenerator(
        vs_mask_path,
        vs_image_path,
        input_size=args.input_size,
        depth_size=args.depth_size,
        transform=input_transform,
        target_transform=transform,
        full_channel_mask=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create model
    print("🧠 Creating model...")
    in_channels = args.num_class_labels
    out_channels = 1
    
    model = create_model(
        args.input_size, 
        args.num_channels, 
        args.num_res_blocks, 
        in_channels=in_channels, 
        out_channels=out_channels
    ).to(device)
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        model,
        image_size=args.input_size,
        depth_size=args.depth_size,
        timesteps=args.timesteps,
        loss_type='l1',
        with_condition=args.with_condition,
        channels=out_channels
    ).to(device)
    
    # Load pretrained weights
    print("📥 Loading pretrained weights...")
    if os.path.exists(model_path):
        weight = torch.load(model_path, map_location=device)
        diffusion.load_state_dict(weight['ema'])
        print("✅ Pretrained model loaded!")
    else:
        print("⚠️  No pretrained model found, training from scratch")
    
    # Create trainer
    print("🏃 Creating trainer...")
    trainer = Trainer(
        diffusion,
        dataset,
        image_size=args.input_size,
        depth_size=args.depth_size,
        train_batch_size=args.batchsize,
        train_lr=args.train_lr,
        train_num_steps=args.epochs,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        fp16=False,
        with_condition=args.with_condition,
        save_and_sample_every=args.save_and_sample_every,
    )
    
    print("🚀 Starting VS fine-tuning...")
    print(f"Training for {args.epochs} epochs with batch size {args.batchsize}")
    trainer.train()
    
    # Upload results to GCS
    print("📤 Uploading results to GCS...")
    if os.path.exists('./results'):
        uploaded_count = 0
        for root, dirs, files in os.walk('./results'):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, './results')
                gcs_path = f"{args.output_path}results/{relative_path}"
                upload_to_gcs(args.bucket_name, local_path, gcs_path)
                uploaded_count += 1
        print(f"✅ Uploaded {uploaded_count} result files")
    else:
        print("⚠️  No results directory found")
    
    print("✅ Training completed and results uploaded!")

if __name__ == "__main__":
    main()

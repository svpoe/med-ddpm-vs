#-*- coding:utf-8 -*-
# +
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import NiftiImageGenerator, NiftiPairImageGenerator
import argparse
import torch

import os 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


parser = argparse.ArgumentParser()
# --- Dataset paths ---
parser.add_argument('-i', '--inputfolder', type=str, default="dataset/vs/mask/")
parser.add_argument('-t', '--targetfolder', type=str, default="dataset/vs/image/")
# --- Data shape and model config ---
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=3)
# --- Training settings ---
parser.add_argument('--train_lr', type=float, default=1e-5)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50000) # epochs parameter specifies the number of training iterations
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--save_and_sample_every', type=int, default=1000)
# --- Fine-tuning options ---
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="model/model_128.pt")
# --- Debug/test/local options ---
parser.add_argument('--cpu_only', action='store_true',
                    help="Force CPU-only execution (for debugging or dry-runs)") #ADDED
parser.add_argument('--dry_run', action='store_true',
                    help="Run one small training loop to validate setup") #ADDED

args = parser.parse_args()

inputfolder = args.inputfolder
targetfolder = args.targetfolder
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
train_lr = args.train_lr

# # Determine device based on arguments #ADDED
# device = 'cpu' if args.cpu_only else ('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}") #ADDED

# # Adjust epochs for dry run #ADDED
# if args.dry_run:
#     args.epochs = 10  # Just a few steps for validation
#     print("DRY RUN MODE: Limited to 10 training steps")

# Define transform functions to avoid pickle errors with lambda
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

# input tensor: (B, 1, H, W, D)  value range: [-1, 1]
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

if with_condition:
    dataset = NiftiPairImageGenerator(
        inputfolder,
        targetfolder,
        input_size=input_size,
        depth_size=depth_size,
        transform=input_transform if with_condition else transform,
        target_transform=transform,
        full_channel_mask=True
    )
else:
    dataset = NiftiImageGenerator(
        inputfolder,
        input_size=input_size,
        depth_size=depth_size,
        transform=transform
    )

print(len(dataset))

in_channels = num_class_labels if with_condition else 1
out_channels = 1

#was .cuda() 
model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda() #to(device)

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    with_condition=with_condition,
    channels=out_channels
).cuda()
#).to(device)   

if len(resume_weight) > 0:
    #weight = torch.load(resume_weight, map_location=device)
    weight = torch.load(resume_weight, map_location='cuda')
    diffusion.load_state_dict(weight['ema'])
    print("Model Loaded!")

trainer = Trainer(
    diffusion,
    dataset,
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = train_lr,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,#True,                       # turn on mixed precision training with apex
    with_condition=with_condition,
    save_and_sample_every = save_and_sample_every,
)

# #ADDED
# # Fix DataLoader num_workers for CPU/dry-run compatibility
# if args.cpu_only or args.dry_run:
#     # Replace the DataLoader to use num_workers=0 for CPU compatibility
#     from torch.utils.data import DataLoader
#     from diffusion_model.trainer import cycle
#     trainer.dl = cycle(DataLoader(trainer.ds, batch_size=args.batchsize, shuffle=True, num_workers=0, pin_memory=False))
#     print("Using DataLoader with num_workers=0 for CPU compatibility")
    
trainer.train()

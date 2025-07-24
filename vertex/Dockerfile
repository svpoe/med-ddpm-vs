# Use Google's Deep Learning container as base
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_cloud.txt .
RUN pip install --no-cache-dir -r requirements_cloud.txt

# Copy the entire codebase
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Default command (will be overridden by Vertex AI)
CMD ["python", "train_vs_cloud.py"]

# Use Google's Deep Learning container as base
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . .

# Set environment variables for proper imports
ENV PYTHONPATH=/app:/app/vertex
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Ensure all Python files are executable
RUN find /app -name "*.py" -exec chmod +x {} \;

# Create necessary directories
RUN mkdir -p /app/dataset /app/model /app/results /app/logs

# Default command (will be overridden by Vertex AI)
CMD ["python", "/app/vertex/train_vs_cloud.py"]

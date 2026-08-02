# ChargeFlow AI V2 — Production Dockerfile
# Python 3.12 slim base image for reproducible lightweight execution
FROM python:3.12-slim

# Set environment variables for Python performance & non-interactive installs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# Install system dependencies required for C extension builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-compatible Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy project source, data, model artifacts, and dashboard application
COPY . .

# Expose Streamlit port 8501 (FastAPI port 8000 can also be mapped)
EXPOSE 8501 8000

# Default command runs the Streamlit UI dashboard
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

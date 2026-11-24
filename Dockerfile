# Multi-stage production Dockerfile for SGRH ImmunoPredict
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501

# Install essential system dependencies (libgomp1 is required by LightGBM / OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application artifacts, serialized models, and demo data
COPY assets/ ./assets/
COPY models/ ./models/
COPY data/raw/ICI.xlsx ./data/raw/ICI.xlsx
COPY pipeline.py .
COPY app.py .

# Healthcheck to monitor Streamlit status
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Expose standard Streamlit port
EXPOSE 8501

# Run Streamlit with production flags
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--theme.base=light"]

FROM python:3.12.8-slim

# Avoid interactive prompts and buffer issues
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# System deps for Pillow/torchvision image ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src ./src
COPY assets ./assets
COPY models ./models

# Expose Streamlit port
EXPOSE 8501

# Default command: launch Streamlit UI
CMD ["streamlit", "run", "src/ui/app.py"]

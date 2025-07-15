# Use a minimal Python image
FROM python:3.10-slim

# Set environment variables to reduce overhead
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Set working directory
WORKDIR /app

# Install system dependencies (needed by some NLP libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port that Cloud Run will connect to
EXPOSE 8080

# Run the app using Gunicorn (App.py, with instance named "app")
CMD ["gunicorn", "-b", "0.0.0.0:8080", "App:app"]
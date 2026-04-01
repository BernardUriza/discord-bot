FROM python:3.14-slim

WORKDIR /app

# System deps for Discord voice (ffmpeg + opus)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libopus0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY insult/ insult/
COPY persona.md .

# Create storage dir for SQLite
RUN mkdir -p storage

CMD ["python", "-m", "insult", "run"]

FROM python:3.14-slim

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY insult/ insult/
COPY persona.md .

# Create storage dir for SQLite
RUN mkdir -p storage

CMD ["python", "-m", "insult", "run"]

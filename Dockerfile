# Base Python image
FROM python:3.12-slim

# Working directory inside container
WORKDIR /app

# Copy production dependencies
COPY requirements-prod.txt .

# Install production dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
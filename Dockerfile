
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first to improve build caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and model configuration.
COPY src/ ./src/
COPY config/ ./config/

# Create a non-root user for runtime security.
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

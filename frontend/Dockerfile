FROM python:3.10-slim

WORKDIR /app

# Copy source modules first
COPY src/ /app/src/

# Install curl untuk testing di dalam container
RUN apt-get update && apt-get install -y curl

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/backend.py .

# Set environment variables
ENV FLASK_APP=backend.py
ENV FLASK_ENV=development
ENV FLASK_DEBUG=1
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Use Flask development server
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
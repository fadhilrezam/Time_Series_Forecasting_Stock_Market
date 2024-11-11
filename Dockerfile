# Gunakan image Python sebagai base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy file requirements.txt
COPY src/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file ke dalam container
COPY src/ .

# Expose ports
EXPOSE 5000 8501

# Command untuk menjalankan main.py
CMD ["python", "-m", "main"]

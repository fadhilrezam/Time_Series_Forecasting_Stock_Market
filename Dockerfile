# Gunakan image Python sebagai base
FROM python:3.10

# Set working directory
WORKDIR /src

# Copy all root folders and files
COPY . /src

# Copy file requirements.txt
COPY src/requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy semua file ke dalam container
COPY src/ .

# Expose ports
EXPOSE 5000 8501

# Command untuk menjalankan main.py
CMD ["python", "-m", "main"]

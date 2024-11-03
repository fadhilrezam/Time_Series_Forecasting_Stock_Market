import subprocess

# Jalankan backend (misalnya Flask atau FastAPI)
backend_process = subprocess.Popen(["python", "backend.py"])  # Sesuaikan dengan nama file backend kamu

# Jalankan frontend (misalnya Streamlit)
frontend_process = subprocess.Popen(["streamlit", "run", "frontend.py"])  # Sesuaikan dengan nama file frontend kamu

# Tunggu hingga kedua proses selesai
backend_process.wait()
frontend_process.wait()
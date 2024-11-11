import subprocess

# Jalankan backend (misalnya Flask atau FastAPI)
# backend_process = subprocess.Popen(["python", "backend.py"])  # Sesuaikan dengan nama file backend kamu
backend_process = subprocess.Popen(['python', 'backend.py', '--port', '5000'])

# Jalankan frontend (misalnya Streamlit)
# frontend_process = subprocess.Popen(["streamlit", "run", "frontend.py"])  # Sesuaikan dengan nama file frontend kamu
frontend_process = subprocess.Popen(['streamlit', 'run', 'frontend.py', '--server.port', '8501'])
# Tunggu hingga kedua proses selesai
backend_process.wait()
frontend_process.wait()
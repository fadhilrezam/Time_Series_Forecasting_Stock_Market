import logging
from datetime import datetime
import os

logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

log_file_name = f'{datetime.now().strftime("%Y-%m-%d")}.log'
log_file_path = os.path.join(logs_path, log_file_name)
print(log_file_path)

logging.basicConfig(
    filename = log_file_path,
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    logging.info('Logging has started')


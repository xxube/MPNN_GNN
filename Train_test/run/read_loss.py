import os
import json
import re

def read_loss_data(directory):
    file_path = os.path.join(base_dir, directory, 'loss_record')
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return {
                'loss': data['2']['loss'][-1],
                'vali_loss': data['2']['vali_loss'][-1],
                'test_loss': data['2']['test_loss'][-1],
            }
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"Error reading loss data for {directory}: {e}")
        return None

current_dir = os.getcwd()
base_dir = os.path.join(current_dir, '..')

# Identify all relevant model directories
model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.endswith('_model')]

# Retrieve and store loss information
loss_data = {}
valid_model_dirs = []
for model_dir in model_dirs:
    if '_CONT' in model_dir:
        continue
    data = read_loss_data(model_dir)
    if data:
        loss_data[model_dir] = data
        valid_model_dirs.append(model_dir)

# Create and sort log file based on loss
log_file = os.path.join(current_dir, '__DONEmodels_loss_log')
with open(log_file, 'w') as log:
    for index, model_dir in enumerate(sorted(valid_model_dirs, key=lambda x: loss_data[x]['loss']), 1):
        study_id = re.findall(r'_#(\d+)_', model_dir)
        study_id = study_id[0] if study_id else ''
        log.write(f"{index}\n")
        log.write(f"{model_dir}\n")
        log.write(f"study {study_id}\n")
        log.write(f"loss {loss_data[model_dir]['loss']}\n")
        log.write(f"vali_loss {loss_data[model_dir]['vali_loss']}\n")
        log.write(f"test_loss {loss_data[model_dir]['test_loss']}\n")

print(f"Log file created: {log_file}")

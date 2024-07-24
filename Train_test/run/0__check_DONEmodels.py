import os
import json
import re

def extract_loss_info(subdir):
    loss_record_path = os.path.join(checkdir, subdir, 'loss_record')
    try:
        with open(loss_record_path, 'r') as input_file:
            loss_record = json.load(input_file)
            return {
                'loss': loss_record['2']['loss'][-1],
                'vali_loss': loss_record['2']['vali_loss'][-1],
                'test_loss': loss_record['2']['test_loss'][-1],
            }
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"Error reading loss information for {subdir}: {e}")
        return None

currentdir = os.getcwd()
checkdir = os.path.join(currentdir, '..')

# Check all model folders in dir
subdir_list = [name for name in os.listdir(checkdir) if os.path.isdir(os.path.join(checkdir, name)) and name.endswith('_model')]

# Extract all loss info from folders
loss_dict = {}
subdir_list_copy = []
for subdir in subdir_list:
    if '_CONT' in subdir:
        continue
    loss_info = extract_loss_info(subdir)
    if loss_info:
        loss_dict[subdir] = loss_info
        subdir_list_copy.append(subdir)

# Sort and record the model performance to log file
log_filename = os.path.join(currentdir, '__DONEmodels_loss_log')
with open(log_filename, 'w') as f:
    for count, subdir in enumerate(sorted(subdir_list_copy, key=lambda x: loss_dict[x]['loss']), 1):
        study_num = re.findall(r'_#(\d+)_', subdir)
        study_num = study_num[0] if study_num else ''
        f.write(f"{count}\n")
        f.write(f"{subdir}\n")
        f.write(f"study {study_num}\n")
        f.write(f"loss {loss_dict[subdir]['loss']}\n")
        f.write(f"vali_loss {loss_dict[subdir]['vali_loss']}\n")
        f.write(f"test_loss {loss_dict[subdir]['test_loss']}\n")

print(f"Log file created: {log_filename}")

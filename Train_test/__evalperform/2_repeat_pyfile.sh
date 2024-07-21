import os
import subprocess

exec_pyfilename = '__gen_model_evalVALI.py'
ref_pyfilename = '_optuna_VALI.py'
additional_lines = '--hide_plt=1'

currentdir = os.getcwd()

# Find all model directories
model_foldername_list = [name for name in os.listdir(currentdir) if os.path.isdir(name) and name.endswith('_model')]
total_num = len(model_foldername_list)

for count, model_foldername in enumerate(model_foldername_list, start=1):
    print(f"Progress: {count} / {total_num}")
    command = f"python {exec_pyfilename} {model_foldername} {ref_pyfilename} {additional_lines}"
    subprocess.run(command, shell=True)

# Optionally remove the temporary log file if created by subprocess
if os.path.exists('_tmp_LOG'):
    os.remove('_tmp_LOG')

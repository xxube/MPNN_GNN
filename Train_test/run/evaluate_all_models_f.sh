import os
import subprocess

def read_model_list(list_file):
    with open(list_file, 'r') as file:
        return file.read().splitlines()

def execute_evaluation(exec_script, model_folder, ref_script, additional_args):
    command = f'python {exec_script} {model_folder} {ref_script} {additional_args} > _tmp_LOG'
    subprocess.run(command, shell=True)

def clean_up_temp_log(log_file='_tmp_LOG'):
    if os.path.exists(log_file):
        os.remove(log_file)

def evaluate_all_models(exec_script, ref_script, additional_args, list_file):
    current_dir = os.getcwd()
    model_folders = read_model_list(list_file)
    
    total_models = len(model_folders)
    
    for index, model_folder in enumerate(model_folders, start=1):
        print(f'Progress: {index} / {total_models}')
        execute_evaluation(exec_script, model_folder, ref_script, additional_args)
    
    clean_up_temp_log()

def main():
    exec_script = '__gen_int_dict.py'
    ref_script = '_optuna_VALI.py'
    additional_args = ''
    list_file = 'modellist'
    
    evaluate_all_models(exec_script, ref_script, additional_args, list_file)

if __name__ == '__main__':
    main()

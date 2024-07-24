import os
import pandas as pd
import shutil
import subprocess

def load_model_list(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
        return df['model_foldername'].tolist()
    else:
        with open(filename, 'r') as file:
            return file.read().splitlines()

def run_evaluation_script(exec_script, model_folder, ref_script, additional_args):
    command = f'python {exec_script} {model_folder} {ref_script} {additional_args} > _tmp_LOG'
    subprocess.run(command, shell=True)

def copy_model_directory(src_dir, dest_dir):
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

def evaluate_models(exec_script, ref_script, additional_args, model_list_file):
    current_dir = os.getcwd()
    ref_script_path = os.path.join(os.path.realpath(os.path.join(current_dir, '..')), ref_script)
    model_list = load_model_list(os.path.join(current_dir, model_list_file))

    total_models = len(model_list)
    for index, model_folder in enumerate(model_list, start=1):
        print(f'Progress: {index} / {total_models}')
        run_evaluation_script(exec_script, model_folder, ref_script_path, additional_args)
        
        src_dir = os.path.join(current_dir, '..', model_folder)
        dest_dir = os.path.join(current_dir, model_folder)
        copy_model_directory(src_dir, dest_dir)

    if os.path.exists('_tmp_LOG'):
        os.remove('_tmp_LOG')

def main():
    exec_script = '../__gen_INT_dict.py'
    ref_script = '_optuna_VALI.py'
    additional_args = ''
    model_list_file = 'modellist'
    
    evaluate_models(exec_script, ref_script, additional_args, model_list_file)

if __name__ == '__main__':
    main()

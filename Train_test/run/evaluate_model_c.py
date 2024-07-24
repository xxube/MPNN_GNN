import shutil
import os
import pandas as pd
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Run model evaluation for each model folder listed in the provided CSV or text file.')
    parser.add_argument('--eval_script', type=str, default='../../__evalperform/__gen_model_evalVALI.py', help='Path to the evaluation script.')
    parser.add_argument('--ref_script', type=str, default='_optuna_VALI.py', help='Reference Python filename.')
    parser.add_argument('--extra_args', type=str, default='--hide_plt=1', help='Additional command line arguments for the evaluation script.')
    parser.add_argument('--model_list_file', type=str, default='modelINFO.csv', help='Filename containing the list of model folder names.')
    return parser.parse_args()

def load_model_list(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df['model_foldername'].tolist()
    else:
        with open(file_path, 'r') as file:
            return file.read().splitlines()

def copy_model(src_dir, dest_dir):
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

def evaluate_model_folders(eval_script, ref_script, extra_args, model_list):
    current_dir = os.getcwd()
    ref_script_path = os.path.join(os.path.realpath(os.path.join(current_dir, '..')), ref_script)
    
    for index, model_folder in enumerate(model_list, start=1):
        print(f'           Progress: {index} / {len(model_list)}')
        os.system(f'python {eval_script} {model_folder} {ref_script_path} {extra_args} > _tmp_LOG')
        copy_model(os.path.join(current_dir, '..', model_folder), os.path.join(current_dir, model_folder))
    
    os.remove('_tmp_LOG')

def main():
    args = get_arguments()
    model_list = load_model_list(args.model_list_file)
    evaluate_model_folders(args.eval_script, args.ref_script, args.extra_args, model_list)

if __name__ == '__main__':
    main()

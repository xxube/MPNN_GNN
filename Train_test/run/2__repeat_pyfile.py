import shutil
import os
import pandas as pd
import argparse

def run_model_evaluation(exec_pyfilename, ref_pyfilename, additional_lines, listfilename):
    currentdir = os.getcwd()
    ref_pyfilename = os.path.join(os.path.realpath(os.path.join(currentdir, '..')), ref_pyfilename)
    
    if os.path.splitext(listfilename)[-1] == '.csv':
        model_info_df = pd.read_csv(os.path.join(currentdir, listfilename))
        model_foldername_list = model_info_df['model_foldername'].tolist()
    else:
        with open(listfilename, 'r') as input_file:
            model_foldername_list = input_file.read().splitlines()
    
    for count, model_foldername in enumerate(model_foldername_list, start=1):
        print(f'           Progress: {count} / {len(model_foldername_list)}')
        os.system(f'python {exec_pyfilename} {model_foldername} {ref_pyfilename} {additional_lines} > _tmp_LOG')
        shutil.copytree(os.path.join(currentdir, '..', model_foldername), os.path.join(currentdir, model_foldername), dirs_exist_ok=True)
    
    os.remove('_tmp_LOG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model evaluation for each model folder listed in the provided CSV or text file.')
    parser.add_argument('--exec_pyfilename', type=str, default='../../__evalperform/__gen_model_evalVALI.py', help='Path to the evaluation script.')
    parser.add_argument('--ref_pyfilename', type=str, default='_optuna_VALI.py', help='Reference Python filename.')
    parser.add_argument('--additional_lines', type=str, default='--hide_plt=1', help='Additional command line arguments for the evaluation script.')
    parser.add_argument('--listfilename', type=str, default='modelINFO.csv', help='Filename containing the list of model folder names.')
    
    args = parser.parse_args()
    
    run_model_evaluation(args.exec_pyfilename, args.ref_pyfilename, args.additional_lines, args.listfilename)

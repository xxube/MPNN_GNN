import os
import pandas as pd
import shutil
import subprocess

def run_model_evaluation(exec_pyfilename, ref_pyfilename, additional_lines, listfilename):
    currentdir = os.getcwd()

    ref_pyfilename = os.path.join(os.path.realpath(os.path.join(currentdir, '..')), ref_pyfilename)

    if os.path.splitext(listfilename)[-1] == '.csv':
        model_info_df = pd.read_csv(os.path.join(currentdir, listfilename))
        model_foldername_list = model_info_df['model_foldername'].tolist()
    else:
        with open(listfilename, 'r') as input_file:
            model_foldername_list = input_file.read().splitlines()

    total_models = len(model_foldername_list)

    for count, model_foldername in enumerate(model_foldername_list, start=1):
        print(f'Progress: {count} / {total_models}')
        command = f'python {exec_pyfilename} {model_foldername} {ref_pyfilename} {additional_lines} > _tmp_LOG'
        subprocess.run(command, shell=True)

        source_dir = os.path.join(currentdir, '..', model_foldername)
        dest_dir = os.path.join(currentdir, model_foldername)
        shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)

    if os.path.exists('_tmp_LOG'):
        os.remove('_tmp_LOG')

if __name__ == '__main__':
    exec_pyfilename = '../__gen_INT_dict.py'
    ref_pyfilename = '_optuna_VALI.py'
    additional_lines = ''
    listfilename = 'modellist'

    run_model_evaluation(exec_pyfilename, ref_pyfilename, additional_lines, listfilename)

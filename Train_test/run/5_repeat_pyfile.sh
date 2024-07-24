import os
import subprocess

def run_model_evaluation(exec_pyfilename, ref_pyfilename, additional_lines, listfilename):
    currentdir = os.getcwd()
    
    # Read model folder names from the list file
    with open(listfilename, 'r') as input_file:
        model_foldername_list = input_file.read().splitlines()
    
    total_models = len(model_foldername_list)
    
    for count, model_foldername in enumerate(model_foldername_list, start=1):
        print(f'Progress: {count} / {total_models}')
        command = f'python {exec_pyfilename} {model_foldername} {ref_pyfilename} {additional_lines} > _tmp_LOG'
        subprocess.run(command, shell=True)
    
    if os.path.exists('_tmp_LOG'):
        os.remove('_tmp_LOG')

if __name__ == '__main__':
    exec_pyfilename = '__gen_int_dict.py'
    ref_pyfilename = '_optuna_VALI.py'
    additional_lines = ''
    listfilename = 'modellist'
    
    run_model_evaluation(exec_pyfilename, ref_pyfilename, additional_lines, listfilename)

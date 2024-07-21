import os
import argparse
import json
import numpy as np
import pandas as pd
from statistics import mean

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modellist', type=str, default=None)
    parser.add_argument('--include_CONT', type=int, default=0)
    parser.add_argument('--mode', type=str, default=None)
    return parser.parse_args()

def get_model_folders(currentdir, modellist, include_CONT):
    if modellist is None:
        model_folder_list = [model_folder for model_folder in os.listdir(currentdir) 
                             if "_model" in model_folder and model_folder[0] != "~"
                             and os.path.isdir(os.path.join(currentdir, model_folder))]
    else:
        with open(os.path.join(currentdir, modellist), "r") as input_file:
            model_folder_list = [line.strip() for line in input_file if line.strip()]
    
    if not include_CONT:
        model_folder_list = [folder for folder in model_folder_list if "_CONT" not in folder]
    
    return model_folder_list

def process_model_folders(currentdir, model_folder_list):
    output_dict = {
        'inputs_E1_Weighted': [],
        'inputs_E2_Weighted': [],
        'y_pred': [],
        'y_true': [],
    }
    
    datapt_totalnum = -1
    
    for model_folder in model_folder_list:
        df = pd.read_csv(os.path.join(currentdir, model_folder, "_INT_dict.csv"))
        datapt_totalnum = len(df)
        
        for key in output_dict.keys():
            output_dict[key].append(df[key].values.tolist())
    
    for key, values in output_dict.items():
        data = np.array(values).T.tolist()
        if len(data) != datapt_totalnum:
            raise ValueError("ERROR in length")
        output_dict[key] = [mean(x) for x in data]
    
    for col in ['data_type', 'tag', 'rct_id', 'lig_id']:
        output_dict[col] = df[col].values.tolist()
    
    return pd.DataFrame.from_dict(output_dict)

def main():
    args = parse_arguments()
    currentdir = os.getcwd()
    
    model_folder_list = get_model_folders(currentdir, args.modellist, args.include_CONT)
    
    output_df = process_model_folders(currentdir, model_folder_list)
    
    output_df.to_csv("avgINT.csv", index=False)
    print("Averaged interaction data saved to avgINT.csv")

if __name__ == "__main__":
    main()

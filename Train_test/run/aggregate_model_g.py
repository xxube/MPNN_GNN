import os
import argparse
import json
import numpy as np
import pandas as pd
from statistics import mean

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modellist', type=str, default=None)
    parser.add_argument('--include_CONT', type=int, default=0)
    parser.add_argument('--mode', type=str, default=None)
    return parser.parse_args()

def list_model_folders(currentdir, modellist, include_CONT):
    if modellist is None:
        model_folders = [folder for folder in os.listdir(currentdir) 
                         if "_model" in folder and folder[0] != "~"
                         and os.path.isdir(os.path.join(currentdir, folder))]
    else:
        with open(os.path.join(currentdir, modellist), "r") as input_file:
            model_folders = [line.strip() for line in input_file if line.strip()]
    
    if not include_CONT:
        model_folders = [folder for folder in model_folders if "_CONT" not in folder]
    
    return model_folders

def aggregate_model_data(currentdir, model_folders):
    data_dict = {
        'inputs_E1_Weighted': [],
        'inputs_E2_Weighted': [],
        'y_pred': [],
        'y_true': [],
    }
    
    total_datapoints = -1
    
    for folder in model_folders:
        df = pd.read_csv(os.path.join(currentdir, folder, "_INT_dict.csv"))
        total_datapoints = len(df)
        
        for key in data_dict.keys():
            data_dict[key].append(df[key].values.tolist())
    
    for key, values in data_dict.items():
        combined_data = np.array(values).T.tolist()
        if len(combined_data) != total_datapoints:
            raise ValueError("ERROR in length")
        data_dict[key] = [mean(x) for x in combined_data]
    
    for col in ['data_type', 'tag', 'rct_id', 'lig_id']:
        data_dict[col] = df[col].values.tolist()
    
    return pd.DataFrame.from_dict(data_dict)

def save_aggregated_data(df):
    df.to_csv("avgINT.csv", index=False)
    print("Averaged interaction data saved to avgINT.csv")

def main():
    args = parse_args()
    currentdir = os.getcwd()
    
    model_folders = list_model_folders(currentdir, args.modellist, args.include_CONT)
    
    aggregated_df = aggregate_model_data(currentdir, model_folders)
    
    save_aggregated_data(aggregated_df)

if __name__ == "__main__":
    main()

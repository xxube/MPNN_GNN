import pandas as pd
import os
import json

def load_initial_dataframe(csv_path):
    return pd.read_csv(csv_path)

def initialize_update_dict(model_foldernames):
    return {'model_foldername': model_foldernames.tolist()}

def process_evaluation_result(eval_result_path, update_dict):
    with open(eval_result_path, 'r') as file:
        perf_dict = json.load(file)
        for datatype, inner_dict in perf_dict.items():
            for key, value in inner_dict.items():
                col_name = f"{datatype}_{key}"
                if col_name in update_dict:
                    update_dict[col_name].append(value)
                else:
                    update_dict[col_name] = [value]

def update_model_info(model_info_df, current_dir):
    update_dict = initialize_update_dict(model_info_df['model_foldername'])
    for model_foldername in model_info_df['model_foldername']:
        print(model_foldername)
        eval_result_path = os.path.join(current_dir, model_foldername, '_eval_result')
        process_evaluation_result(eval_result_path, update_dict)
    return pd.DataFrame(update_dict)

def main():
    current_dir = os.getcwd()
    model_info_csv = os.path.join(current_dir, 'modelINFO.csv')
    model_info_df = load_initial_dataframe(model_info_csv)
    updated_df = update_model_info(model_info_df, current_dir)
    final_df = pd.concat([model_info_df, updated_df], axis=1)
    final_df.to_csv(os.path.join(current_dir, 'modelINFO_updated.csv'), index=False)

if __name__ == '__main__':
    main()

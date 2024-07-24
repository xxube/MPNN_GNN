import os
import re
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Top10', type=str, default='0')
    parser.add_argument('--Top1', type=str, default='0')
    parser.add_argument('--print_zip', type=str, default='0')
    return parser.parse_args()

def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def extract_model_info(log_content):
    model_info = {
        'model_foldername': [],
        'study': [],
        'loss': [],
        'vali_loss': [],
        'test_loss': [],
    }
    model_entries = re.split(r'\n\d+\n', log_content[2:])
    for entry in model_entries:
        entry_lines = [line for line in entry.split('\n') if line]
        if len(entry_lines) != 5:
            print(f"Error processing entry: {entry_lines}")
            continue
        model_info['model_foldername'].append(entry_lines[0])
        for line in entry_lines[1:]:
            key, value = line.split(' ')
            model_info[key].append(float(value) if value else 0.0)
    return model_info

def create_dataframe(model_info):
    df = pd.DataFrame.from_dict(model_info)
    return df.sort_values(['vali_loss'], ascending=True)

def filter_dataframe(df, top10, top1):
    if top10:
        return df.head(10)
    elif top1:
        return df.head(1)
    else:
        return df[df['vali_loss'] < 2.0]

def print_zip_command(df):
    filtered_models = df['model_foldername'].tolist()
    print(f"zip -r modelpkg.zip {' '.join(filtered_models)}")

def save_to_csv(df, file_path):
    df.to_csv(file_path, index=False)

def main():
    args = parse_arguments()
    current_dir = os.getcwd()
    log_file_path = os.path.join(current_dir, '__DONEmodels_loss_log')
    
    log_content = read_log_file(log_file_path)
    model_info = extract_model_info(log_content)
    model_df = create_dataframe(model_info)
    
    filtered_df = filter_dataframe(model_df, args.Top10 == '1', args.Top1 == '1')
    
    if args.print_zip == '1':
        print_zip_command(filtered_df)
    
    csv_file_path = os.path.join(current_dir, 'modelINFO.csv')
    save_to_csv(filtered_df, csv_file_path)

if __name__ == "__main__":
    main()

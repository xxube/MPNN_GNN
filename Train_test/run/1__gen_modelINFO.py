import os
import re
import pandas as pd
import argparse

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--Top10', type=str, default='0')
parser.add_argument('--Top1', type=str, default='0')
parser.add_argument('--print_zip', type=str, default='0')
args = parser.parse_args()
_Top10 = args.Top10 == '1'
_Top1 = args.Top1 == '1'
_print_zip = args.print_zip == '1'

currentdir = os.getcwd()

# Read the models' loss log
with open(os.path.join(currentdir, '__DONEmodels_loss_log'), 'r') as input_file:
    log_content = input_file.read()

# Initialize dictionary to hold model information
modelINFO_dict = {
    'model_foldername': [],
    'study': [],
    'loss': [],
    'vali_loss': [],
    'test_loss': [],
}

# Separate all models
model_entries = re.split(r'\n\d+\n', log_content[2:])
for entry in model_entries:
    entry_lines = [line for line in entry.split('\n') if line]
    if len(entry_lines) != 5:
        print(f"Error processing entry: {entry_lines}")
        continue
    modelINFO_dict['model_foldername'].append(entry_lines[0])
    for line in entry_lines[1:]:
        key, value = line.split(' ')
        modelINFO_dict[key].append(float(value) if value else 0.0)

# Convert dictionary to DataFrame
modelINFO_df = pd.DataFrame.from_dict(modelINFO_dict)
modelINFO_df = modelINFO_df.sort_values(['vali_loss'], ascending=True)

# Filter DataFrame based on arguments
if _Top10:
    modelINFO_df = modelINFO_df.head(10)
elif _Top1:
    modelINFO_df = modelINFO_df.head(1)
else:
    modelINFO_df = modelINFO_df[modelINFO_df['vali_loss'] < 2.0]

# Print command to zip the filtered models if requested
if _print_zip:
    filtered_model_list = modelINFO_df['model_foldername'].tolist()
    print(f"zip -r modelpkg.zip {' '.join(filtered_model_list)}")

# Save DataFrame to CSV
modelINFO_df.to_csv(os.path.join(currentdir, 'modelINFO.csv'), index=False)

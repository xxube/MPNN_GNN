import pandas as pd
import os
import json

# Get the current working directory
currentdir = os.getcwd()

# Load the initial DataFrame from CSV
modelINFO_df = pd.read_csv(os.path.join(currentdir, 'modelINFO.csv'))

# Initialize a dictionary to hold new data
update_dict = {'model_foldername': []}

# Iterate over each model folder
for model_foldername in modelINFO_df['model_foldername']:
    print(model_foldername)
    
    # Append the model folder name to the update dictionary
    update_dict['model_foldername'].append(model_foldername)
    
    # Path to the evaluation result file
    eval_result_path = os.path.join(currentdir, model_foldername, '_eval_result')
    
    # Read and process the evaluation result JSON file
    with open(eval_result_path, 'r') as input_file:
        perf_dict = json.load(input_file)
        for datatype, inner_dict in perf_dict.items():
            for k, v in inner_dict.items():
                # Construct the column name
                update_colname = f"{datatype}_{k}"
                
                # Update the dictionary with new data
                if update_colname in update_dict:
                    update_dict[update_colname].append(v)
                else:
                    update_dict[update_colname] = [v]

# Convert the update dictionary to a DataFrame
update_df = pd.DataFrame(update_dict)

# Concatenate the original and updated DataFrames
modelINFO_df = pd.concat([modelINFO_df, update_df], axis=1)

# Save the updated DataFrame to a new CSV file
modelINFO_df.to_csv(os.path.join(currentdir, 'modelINFO_updated.csv'), index=False)

import os
import pandas as pd
import argparse

def load_model_info(file_path):
    return pd.read_csv(file_path)

def filter_model_data(df, top10, top1, vali_loss_threshold, valid_r_square_threshold, train_r_square_threshold):
    if top10:
        return df.nsmallest(10, 'vali_loss')
    elif top1:
        return df.nsmallest(1, 'vali_loss')
    else:
        return df[
            (df['vali_loss'] < vali_loss_threshold) &
            (df['valid_R_square'] > valid_r_square_threshold) &
            (df['train_R_square'] > train_r_square_threshold)
        ]

def save_filtered_models(filtered_df, output_path):
    with open(output_path, 'w') as output_file:
        output_file.write("\n".join(filtered_df['model_foldername'].values.tolist()))
    print(f"Filtered model list saved to {output_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter models based on validation loss and R-square values.")
    parser.add_argument('--Top10', action='store_true', help="Select the top 10 models based on validation loss.")
    parser.add_argument('--Top1', action='store_true', help="Select the top 1 model based on validation loss.")
    parser.add_argument('--vali_loss', type=float, default=1.5, help="Validation loss threshold.")
    parser.add_argument('--valid_R_square', type=float, default=0.8, help="Validation R-square threshold.")
    parser.add_argument('--train_R_square', type=float, default=0.0, help="Training R-square threshold.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    current_dir = os.getcwd()
    model_info_path = os.path.join(current_dir, 'modelINFO_updated.csv')
    output_filename = 'modellist'

    model_info_df = load_model_info(model_info_path)
    filtered_df = filter_model_data(
        model_info_df, args.Top10, args.Top1, args.vali_loss,
        args.valid_R_square, args.train_R_square
    )
    output_path = os.path.join(current_dir, output_filename)
    save_filtered_models(filtered_df, output_path)

if __name__ == "__main__":
    main()

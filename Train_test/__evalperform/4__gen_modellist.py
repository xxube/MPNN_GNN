import os
import pandas as pd
import argparse

def filter_models(
    model_info_path='modelINFO_updated.csv',
    output_filename='modellist',
    top10=False,
    top1=False,
    vali_loss_threshold=1.5,
    valid_r_square_threshold=0.8,
    train_r_square_threshold=0.0
):
    currentdir = os.getcwd()

    # Read the updated model information
    model_info_df = pd.read_csv(os.path.join(currentdir, model_info_path))

    # Apply filtering based on provided thresholds
    if top10:
        filtered_df = model_info_df.nsmallest(10, 'vali_loss')
    elif top1:
        filtered_df = model_info_df.nsmallest(1, 'vali_loss')
    else:
        filtered_df = model_info_df[
            (model_info_df['vali_loss'] < vali_loss_threshold) &
            (model_info_df['valid_R_square'] > valid_r_square_threshold) &
            (model_info_df['train_R_square'] > train_r_square_threshold)
        ]

    # Write the filtered model folder names to the output file
    output_path = os.path.join(currentdir, output_filename)
    with open(output_path, 'w') as output_file:
        output_file.write("\n".join(filtered_df['model_foldername'].values.tolist()))

    print(f"Filtered model list saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter models based on validation loss and R-square values.")
    parser.add_argument('--Top10', action='store_true', help="Select the top 10 models based on validation loss.")
    parser.add_argument('--Top1', action='store_true', help="Select the top 1 model based on validation loss.")
    parser.add_argument('--vali_loss', type=float, default=1.5, help="Validation loss threshold.")
    parser.add_argument('--valid_R_square', type=float, default=0.8, help="Validation R-square threshold.")
    parser.add_argument('--train_R_square', type=float, default=0.0, help="Training R-square threshold.")
    args = parser.parse_args()

    filter_models(
        top10=args.Top10,
        top1=args.Top1,
        vali_loss_threshold=args.vali_loss,
        valid_r_square_threshold=args.valid_R_square,
        train_r_square_threshold=args.train_R_square
    )

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.offsetbox import AnchoredText

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_filename', type=str)
    parser.add_argument('--hide_plt', type=int, default=0)
    parser.add_argument('--round_digits', type=int, default=2)
    args = parser.parse_args()
    args.result_filename = args.result_filename.replace('.csv', '') + '.csv'
    return args

def generate_plot(x, y, set_type, round_digits, save_dir, hide_plot):
    plt.rc('xtick', labelsize=32)
    plt.rc('ytick', labelsize=32)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    linreg = LinearRegression().fit(x.reshape(-1, 1), y)
    linreg.coef_ = np.array([1])
    linreg.intercept_ = 0
    r_square = linreg.score(x.reshape(-1, 1), y)

    plt.scatter(x, y, s=10)

    plot_min = 0
    plot_max = 50
    ax.plot([plot_min - 1, plot_max + 1], [plot_min - 1, plot_max + 1], 'k--', lw=1.75)
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)

    plt.xlabel('Experimental rate constant\n k (1000/h⁻¹)', fontsize=36)
    plt.ylabel('Predicted rate constant\n k (1000/h⁻¹)', fontsize=36)

    legend_text = f'R²    = {round(r_square, round_digits) if r_square >= 0 else "< 0"}'
    legend_text += f'\nMAE = {round(mean_absolute_error(y, x), round_digits)}'
    legend_text += f'\nRMS = {round(np.sqrt(mean_squared_error(y, x)), round_digits)}'
    at = AnchoredText(legend_text, prop=dict(size=36), frameon=True, loc='lower right')
    ax.add_artist(at)

    fig.text(-0.05, 0.95, {'train': 'a)', 'vali': 'b)', 'test': 'c)'}[set_type],
             horizontalalignment='left', verticalalignment='top', size=45)
    plt.title({'train': 'Training Set Performance', 'vali': 'Validation Set Performance', 'test': 'Testing Set Performance'}[set_type],
              size=35, pad=25)

    plt.savefig(os.path.join(save_dir, f'{set_type}.png'), bbox_inches='tight', dpi=500)

    if not hide_plot:
        plt.show()

    return r_square

def generate_combined_plot(x_dict, y_dict, save_dir, hide_plot):
    plt.rc('xtick', labelsize=32)
    plt.rc('ytick', labelsize=32)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    scatter_size = 15
    for set_type, x in x_dict.items():
        y = y_dict[set_type]
        if set_type == "train":
            train_scat = plt.scatter(x, y, s=scatter_size, c='red', marker='o')
        elif set_type == "vali":
            vali_scat = plt.scatter(x, y, s=scatter_size, c='green', marker='x')
        elif set_type == "test":
            test_scat = plt.scatter(x, y, s=scatter_size, c='blue', marker='^')
    
    lgnd = plt.legend((train_scat, vali_scat, test_scat),
                      ('Training set', 'Validation set', 'Testing set'),
                      scatterpoints=1,
                      loc='lower right',
                      fontsize=20)
    for i in range(3):
        lgnd.legendHandles[i]._sizes = [50]

    plot_min = 0
    plot_max = 50
    ax.plot([plot_min - 1, plot_max + 1], [plot_min - 1, plot_max + 1], 'k--', lw=1.75)

    plt.xlabel("Experimental rate constant\n k (1000/h⁻¹)", fontsize=36)
    plt.ylabel("Predicted rate constant\n k (1000/h⁻¹)", fontsize=36)

    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)

    title = 'Performance of \nTraining/Validation/Testing Sets'
    plt.title(title, size=35, pad=25)

    plt.savefig(os.path.join(save_dir, "ALL.png"), bbox_inches="tight", dpi=450)

    if not hide_plot:
        plt.show()

def save_results(result, save_dir):
    with open(os.path.join(save_dir, 'avgPerform_LOG'), 'w') as output:
        json.dump(result, output, indent=4)

def main():
    args = parse_args()

    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'avgPerform')
    os.makedirs(save_dir, exist_ok=True)

    input_df = pd.read_csv(os.path.join(current_dir, args.result_filename))

    results = {set_type: {'values': input_df[input_df['data_type'] == set_type]['y_true'].values.tolist(),
                          'pred': input_df[input_df['data_type'] == set_type]['y_pred'].values.tolist()}
               for set_type in ['train', 'vali', 'test']}

    all_x_dict, all_y_dict = {}, {}
    for set_type in ['train', 'vali', 'test']:
        y = np.array(results[set_type]['values'])
        y_pred = np.array(results[set_type]['pred'])

        results[set_type]['MAE'] = mean_absolute_error(y, y_pred)
        results[set_type]['RMS'] = np.sqrt(mean_squared_error(y, y_pred))
        results[set_type]['R_square'] = generate_plot(y, y_pred, set_type, args.round_digits, save_dir, args.hide_plt)

        all_x_dict[set_type] = y
        all_y_dict[set_type] = y_pred

    generate_combined_plot(all_x_dict, all_y_dict, save_dir, args.hide_plt)

    save_results(results, save_dir)

if __name__ == "__main__":
    main()

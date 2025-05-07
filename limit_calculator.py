import argparse
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.formula.api import ols
from gdrive import Gdrive_control
from scipy.stats import norm
from datetime import datetime

OUTPUT_FILE_PATH = "GRR/Output/limit_calc_outputfile.csv"
METRICS_FILE = "GRR/GRRRawData/X10_Gen2_Cell_TestLimits_2025_04_24.csv" # Declare here the CSV will all the metrics

# Update these values based on your own estimation of a proper tolerance
TOL_MIN = 1
TOL_MAX = 20
MIN_ELEMENTS = 40
# Update for tour target CPK
TARGET_CPK = 1.0
# Update this to somewhere between 20 and 40 depending on how granular your data is.Less bins will have less resolution of data
BINS = 20


def read_file(file_to_read:str) -> pd.DataFrame:
    df = pd.read_csv(file_to_read)    
    return df

def best_limits(test_key_df:pd.DataFrame, mu:float, key_metric:str, cpk_target:float=1.3, file_output:str = OUTPUT_FILE_PATH) -> tuple:

    data_vals = test_key_df['value_double'].to_numpy()
    best_cpk = 0
    best_limits = (None, None)
    #mu = np.mean(data_vals) mu comes from the PDF peak but it also can come from the mean of the dataset
    sigma = np.std(data_vals)
    
    for width in np.linspace(TOL_MIN, TOL_MAX, (TOL_MAX*2)):
        lsl = mu - width / 2
        usl = mu + width / 2
        cpk = min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma))
        if cpk > best_cpk:
            best_cpk = cpk
            best_limits = (lsl, usl)
        # Exit if the target CPK was achieved, if not keep going
        if cpk >= cpk_target:
            break
    save_data(key_metric,"Calculated best Low Limit",best_limits[0], file_output)
    save_data(key_metric,"Calculated best High Limit",best_limits[1], file_output)
    save_data(key_metric,"Peak of the PDF",mu, file_output)
    save_data(key_metric,"CPK estimated",best_cpk, file_output)
    
    return (best_limits[0], best_limits[1])

def save_data(test_key:str, string_to_save:str, value:float, file_output:str=OUTPUT_FILE_PATH):

    with open(file_output, 'a') as out_file:
        out_file.write(f"{test_key},{string_to_save},{value:.2f}\n")
        print(f"{test_key}\t{string_to_save}\t{value}\n")
    return


def create_output_folders(output_dir) -> None:
    try:
        # create folders to store outputs, files and charts
        Path.mkdir(output_dir)
        Path.mkdir(f"{output_dir}charts/")
    except:
        return

    return



def main(save_gsheet:bool):
    
    directory_path = Path('GRR/GRRRawData/')
    files = [f.name for f in directory_path.iterdir() if f.is_file()]
    csv_files = [f for f in files if ("csv" in f)]

    for csv_file in csv_files:

        print(f"Executing analysis for: {csv_file}")
        today_str = datetime.today().strftime('%d.%m.%Y')
        output_dir = f"GRR/Output/{csv_file.strip('.csv')}/"
        create_output_folders(output_dir)
        # store the output file w/ calculations offline
        out_file_path = f"{output_dir}limit_calc_outputfile_{today_str}.csv"
        # Handle the Gdrive control to push a new file to GDRIVE
        if save_gsheet:
            gdrive = Gdrive_control(f"TLA_{csv_file.strip('.csv')}_{today_str}")

        with open(out_file_path, "w") as out_file:
            out_file.write("Test Key,Parameter,Value\n")

        # Filter for a single test at a time (e.g., test1)
        df = read_file(f"GRR/GRRRawData/{csv_file}")
        test_keys = df['key'].unique()
        # Analyze each key in the file
        for test_name in test_keys:
            df_test = df[df['key'] == test_name]

            fig, ax = plt.subplots(figsize=(10, 5))
            # Create secondary y-axis
            ax2 = ax.twinx()
            # Plot histogram
            counts, bins, patches = ax.hist(df_test['value_double'], bins=BINS, alpha=0.7, density=False, label=f'Limit Analysis {test_name}')
            # Fit a normal distribution to the data
            mu, std = norm.fit(df_test['value_double'])
            # Plot the PDF
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax2.plot(x, p, linewidth=2, label='Normal PDF')
            # Vertical line at PDF peak
            max_pdf_x = x[np.argmax(p)]
            ax2.axvline(max_pdf_x, color='r', linestyle='--', linewidth=2, label='PDF Peak')
            low_lim, high_lim = best_limits(test_key_df=df_test, mu=max_pdf_x, key_metric=test_name, cpk_target=TARGET_CPK, file_output=out_file_path)
            ax2.axvline(low_lim, color='b', linestyle='--', linewidth=2, label='Estimated LL')
            ax2.axvline(high_lim, color='b', linestyle='--', linewidth=2, label='Estimated HL')
            ax2.legend()
            ax.set_xlabel('Metric Value')
            ax.set_ylabel('Binarized Metric Count')
            ax2.set_ylabel('Distribution Density')
            plt.title(f'Metric {test_name} Distribution and Estimated Limits')
            plt.savefig(f'{output_dir}charts/{test_name}_{today_str}_limits.png')
            plt.close(fig)
            # Save the test key results to GDrive
            if save_gsheet:
                gdrive.write_to_new_sheet(df_test, test_name[:29])

        # Save GRR analysis as Google Sheets file
        if save_gsheet:
            analysis_df = pd.read_csv(out_file_path)
            gdrive.write_to_new_sheet(analysis_df, "Test Limit Calculations")
            print("Waiting to not hit the quota!")
            gdrive.wait_time(15)

    return


if __name__ == "__main__":
    # Optional flag to push the output to GSheets
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsheets", action="store_true",  help="Save ouputs to GSHEETS" )
    args = parser.parse_args()
    main(save_gsheet=args.gsheets)
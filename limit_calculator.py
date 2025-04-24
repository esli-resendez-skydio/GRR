import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
#from GRR.gdrive import Gdrive_control
from scipy.stats import norm
from datetime import datetime

OUTPUT_FILE_PATH = "GRR/Output/limit_calc_outputfile.csv"
METRICS_FILE = "GRR/GRRRawData/X10_Gen2_Cell_TestLimits_2025_04_24.csv" # Declare here the CSV will all the metrics

def read_file() -> pd.DataFrame:
    df = pd.read_csv(METRICS_FILE)    
    return df

def best_limits(test_key_df:pd.DataFrame, mu:float, key_metric:str, cpk_target:float=1.3, file_output:str = OUTPUT_FILE_PATH) -> tuple:

    data_vals = test_key_df['value_double'].to_numpy()
    best_cpk = 0
    best_limits = (None, None)
    #mu = np.mean(data_vals) mu comes from the PDF peak but it also can come from the mean of the dataset
    sigma = np.std(data_vals)
    
    for width in np.linspace(3, 20, 100):
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




def main():
    
    #gdrive = Gdrive_control("GRR_Analysis_date")
    today_str = datetime.today().strftime('%d.%m.%Y')
    out_file_path = f"GRR/Output/limit_calc_outputfile_{today_str}.csv"


    with open(out_file_path, "w") as out_file:
        out_file.write("Test Key,Parameter,Value\n")

    # Filter for a single test at a time (e.g., test1)
    df = read_file()
    test_keys = df['key'].unique()
    
    for test_name in test_keys:
        df_test = df[df['key'] == test_name]

        fig, ax = plt.subplots(figsize=(10, 5))
        # Create secondary y-axis
        ax2 = ax.twinx()
        # Plot histogram
        counts, bins, patches = ax.hist(df_test['value_double'], bins=30, alpha=0.7, density=False, label=f'Limit Analysis {test_name}')
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
        low_lim, high_lim = best_limits(test_key_df=df_test, mu=max_pdf_x, key_metric=test_name, cpk_target=1.3, file_output=out_file_path)
        ax2.axvline(low_lim, color='b', linestyle='--', linewidth=2, label='Low Limit')
        ax2.axvline(high_lim, color='b', linestyle='--', linewidth=2, label='High Limit')
        ax2.legend()

        plt.title(f'{test_name} Distribution and Estimated Limits')
        plt.savefig(f'GRR/ChartsLimitsCalculator/{test_name}_{today_str}_limits.png')
        plt.close(fig)

    # Save GRR analysis as Google Sheets file
    #push_gsheets()

    return





if __name__ == "__main__":
    main()

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from GRR.gdrive import Gdrive_control

OUTPUT_FILE_PATH = "GRR/Output/grr_outputfile.csv"

def read_file() -> pd.DataFrame:
    df = pd.read_csv('GRR/GRRRawData/Pine_GRR_Mar2025.csv')    
    return df

def best_limits(test_key_df:pd.DataFrame, key_metric:str, cpk_target:float=1.3) -> None:

    data_vals = test_key_df['value_double'].to_numpy()
    best_cpk = 0
    best_limits = (None, None)
    mu = np.mean(data_vals)
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
    save_data(key_metric,"Calculated best Low Limit",best_limits[0])
    save_data(key_metric,"Calculated best High Limit",best_limits[1])
    save_data(key_metric,"Mean (calculated center)",mu)
    save_data(key_metric,"CPK estimated",best_cpk)
    
    return

def save_data(test_key:str, string_to_save:str, value:float):

    with open(OUTPUT_FILE_PATH, 'a') as out_file:
        out_file.write(f"{test_key},{string_to_save},{value:.2f}\n")
        print(f"{test_key}\t{string_to_save}\t{value}\n")
    return




def main():
    
    #gdrive = Gdrive_control("GRR_Analysis_date")

    with open(OUTPUT_FILE_PATH, "w") as out_file:
        out_file.write("Test Key,Parameter,Value\n")

    # Filter for a single test at a time (e.g., test1)
    df = read_file()
    test_keys = df['key'].unique()
    
    for test_name in test_keys:
        df_test = df[df['key'] == test_name]

        # Perform ANOVA: measurement ~ C(part) + C(fixture)
        model = ols('value_double ~ C(serial_number) + C(station_hostname)', data=df_test).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
        anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
        MS_part = anova_table.loc['C(serial_number)', 'mean_sq']
        MS_fixture = anova_table.loc['C(station_hostname)', 'mean_sq']
        MS_repeatability = anova_table.loc['Residual', 'mean_sq']

        # Number of fixtures and parts
        n_fixture = df_test['station_hostname'].nunique()
        n_part = df_test['serial_number'].nunique()

        # Variance components
        var_repeatability = MS_repeatability
        var_reproducibility = (MS_fixture - MS_repeatability) / n_part
        var_part_to_part = (MS_part - MS_repeatability) / n_fixture

        # Total variance
        var_total = var_repeatability + var_reproducibility + var_part_to_part

        # GR&R metrics
        percent_repeatability = 100 * var_repeatability / var_total
        percent_reproducibility = 100 * var_reproducibility / var_total
        percent_grr = percent_repeatability + percent_reproducibility
        save_data(test_name,"Repeatability %",percent_repeatability)
        save_data(test_name,"Reproducibility %:",percent_reproducibility)
        save_data(test_name,"Total GR&R %",percent_grr)
        best_limits(df_test, test_name, cpk_target=1.3)
        # Chart of variation
        # Group data by fixture

        fixtures = df_test['station_hostname'].unique()
        data_by_fixture = [df_test[df_test['station_hostname'] == f]['value_double'].values for f in fixtures]

        plt.figure(figsize=(8, 6))
        plt.boxplot(data_by_fixture, tick_labels=fixtures)
        plt.title(f'{test_name} Dist by Fixture')
        plt.xlabel('Fixture')
        plt.ylabel('Measurement Value')
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'GRR/Charts/{test_name}_boxplot.png')

        plt.figure(figsize=(8, 6))
        plt.pie([var_repeatability, var_reproducibility, var_part_to_part], labels=['Repeatability', 'Reproducibility', 'Part-to-part'])
        plt.title(f'GRR Data {test_name}')
        plt.savefig(f'GRR/Charts/{test_name}_pie.png')

    # Save GRR analysis as Google Sheets file
    #push_gsheets()

    return





if __name__ == "__main__":
    main()

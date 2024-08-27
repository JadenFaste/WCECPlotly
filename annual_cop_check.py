import pandas as pd
import matplotlib.pyplot as plt

# Frontier cycle data
path_cycle_2022 = r"C:\Users\cdgreen\OneDrive - University of California, Davis\Documents\Villara_MFHP\dashboard\tango_site\data\Villara 3 Function Cycle Data - 2022.csv"
path_cycle_2023 = r"C:\Users\cdgreen\OneDrive - University of California, Davis\Documents\Villara_MFHP\dashboard\tango_site\data\Villara 3 Function Cycle Data - 2023.csv"
path_cycle_2024 = r"C:\Users\cdgreen\OneDrive - University of California, Davis\Documents\Villara_MFHP\dashboard\tango_site\data\2024-08-12 Villara 3 Function Cycle Data - 2024.csv"

df_cycle_2022 = pd.read_csv(path_cycle_2022)
df_cycle_2023 = pd.read_csv(path_cycle_2023)
df_cycle_2024 = pd.read_csv(path_cycle_2024)

df_cycle_2022.index = pd.to_datetime(df_cycle_2022['idx_Mid_Cycle'])
df_cycle_2023.index = pd.to_datetime(df_cycle_2023['idx_Mid_Cycle'])
df_cycle_2024.index = pd.to_datetime(df_cycle_2024['idx_Mid_Cycle'])

# Plot COP vs outdoor temperature
def ac_mode_cop_plot(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end):

    df_cycle_2022 = df_cycle_2022[df_cycle_2022['AC_Mode'] == 1]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['AC_Mode'] == 1]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['AC_Mode'] == 1]
    
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['AC_and_DHW_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['AC_and_DHW_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['AC_and_DHW_Mode'] == 0]
    
    df_cycle_2024 = df_cycle_2024.loc['2024-07-19':]
    df_cycle_latest = df_cycle_2024.loc[latest_begin:latest_end]
    
    plt.figure()
    plt.scatter(df_cycle_2022['T_Outdoor_ecobee_F'],df_cycle_2022['COP_Steady_State'],label = '2022',alpha=0.5)
    plt.scatter(df_cycle_2023['T_Outdoor_ecobee_F'],df_cycle_2023['COP_Steady_State'], label = '2023',alpha=0.5)
    plt.scatter(df_cycle_2024['T_Outdoor_ecobee_F'],df_cycle_2024['COP_Steady_State'], label = '2024',alpha=0.5)
    plt.scatter(df_cycle_latest['T_Outdoor_ecobee_F'],df_cycle_latest['COP_Steady_State'], label = 'Current Selection',edgecolor='black')
    
    plt.legend()
    plt.grid()
    plt.ylim([0,6])
    plt.ylabel('Steady State COP')
    plt.xlabel('Outdoor Air Temperature [°F]')
    plt.title('AC Mode')

    
if __name__ == '__main__':

    latest_begin = '2024-08-10'
    latest_end = '2024-08-11'
    
    ac_mode_cop_plot(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end)    

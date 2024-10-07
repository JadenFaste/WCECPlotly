import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

path = r'C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly'
os.chdir(path)


# Set latest date range
latest_begin = '2024-09-10'
latest_end = '2024-09-26'

#%% Get 5-minute data
def get_data(path):
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df['Unnamed: 0'])
    df = df.resample('5Min').mean()
    return df
    
path_5min_2022 = r"Villara 3 Function 5 Minute Data - 2022.csv"
path_5min_2023 = r"Villara 3 Function 5 Minute Data - 2023.csv"
path_5min_2024 = r"Villara 3 Function 5 Minute Data - 2024.csv"

df_5min_2022 = get_data(path_5min_2022)
df_5min_2023 = get_data(path_5min_2023)
df_5min_2024 = get_data(path_5min_2024)

#%% Get cycle data
path_cycle_2022 = r"Villara 3 Function Cycle Data - 2022.csv"
path_cycle_2023 = r"Villara 3 Function Cycle Data - 2023.csv"
path_cycle_2024 = r"Villara 3 Function Cycle Data - 2024.csv"

df_cycle_2022 = pd.read_csv(path_cycle_2022)
df_cycle_2023 = pd.read_csv(path_cycle_2023)
df_cycle_2024 = pd.read_csv(path_cycle_2024)

df_cycle_2022.index = pd.to_datetime(df_cycle_2022['idx_Mid_Cycle'])
df_cycle_2023.index = pd.to_datetime(df_cycle_2023['idx_Mid_Cycle'])
df_cycle_2024.index = pd.to_datetime(df_cycle_2024['idx_Mid_Cycle'])

#%% Histogram of Outdoor Unit (ODU) Power

# HVAC Cooling Mode ODU Power
def cool_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end):
    
    df_latest = df_cycle_2024.loc[latest_begin:latest_end]
    
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['AC_Mode'] == 1]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['AC_Mode'] == 1]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['AC_Mode'] == 1]
    df_latest = df_latest[df_latest['AC_Mode'] == 1]
    
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['AC_and_DHW_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['AC_and_DHW_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['AC_and_DHW_Mode'] == 0]
    df_latest = df_latest[df_latest['AC_and_DHW_Mode'] == 0]

    plt.figure()
    if len(df_cycle_2022) > 0:
        plt.hist(df_cycle_2022['EP_OutdoorUnit_PowerSum_W'],alpha=0.5,label='2022',color='blue')
    if len(df_cycle_2023) > 0:
        plt.hist(df_cycle_2023['EP_OutdoorUnit_PowerSum_W'],alpha=0.5,label='2023',color='red')
    if len(df_cycle_2024) > 0:
        plt.hist(df_cycle_2024['EP_OutdoorUnit_PowerSum_W'],alpha=0.5,label='2024',color='green')

    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    
    plt.axvline(df_cycle_2022['EP_OutdoorUnit_PowerSum_W'].mean(), color='blue', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+50, max_ylim*0.9, '2022 Mean: {:.2f}'.format(df_cycle_2022['EP_OutdoorUnit_PowerSum_W'].mean()),color='blue')
    
    plt.axvline(df_cycle_2023['EP_OutdoorUnit_PowerSum_W'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+50, max_ylim*0.8, '2023 Mean: {:.2f}'.format(df_cycle_2023['EP_OutdoorUnit_PowerSum_W'].mean()),color='red')
   
    plt.axvline(df_cycle_2024['EP_OutdoorUnit_PowerSum_W'].mean(), color='green', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+50, max_ylim*0.7, '2024 Mean: {:.2f}'.format(df_cycle_2024['EP_OutdoorUnit_PowerSum_W'].mean()),color='green')

    plt.text(min_xlim+50, max_ylim*0.6, 'Latest Mean: {:.2f}'.format(df_latest['EP_OutdoorUnit_PowerSum_W'].mean()),color='black')

    plt.legend(loc='upper right')
    plt.xlabel('Outdoor Unit Power [W]')
    plt.ylabel('Count')
    plt.title('Space Cooling Mode')

cool_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end)

# HVAC Heating Mode ODU Power
def heat_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end):
    
    df_latest = df_cycle_2024.loc[latest_begin:latest_end]
        
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['Space_Heat_Mode'] == 1]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['Space_Heat_Mode'] == 1]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['Space_Heat_Mode'] == 1]
    df_latest = df_latest[df_latest['Space_Heat_Mode'] == 1]
    
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['Defrost_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['Defrost_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['Defrost_Mode'] == 0]
    df_latest = df_latest[df_latest['Defrost_Mode'] == 0]

    plt.figure()
    if len(df_cycle_2022) > 0:
        plt.hist(df_cycle_2022['EP_OutdoorUnit_PowerSum_W'],alpha=0.5,label='2022',color='blue')
    if len(df_cycle_2023) > 0:
        plt.hist(df_cycle_2023['EP_OutdoorUnit_PowerSum_W'],alpha=0.5,label='2023',color='red')
    if len(df_cycle_2024) > 0:
        plt.hist(df_cycle_2024['EP_OutdoorUnit_PowerSum_W'],alpha=0.5,label='2024',color='green')

    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    plt.axvline(df_cycle_2022['EP_OutdoorUnit_PowerSum_W'].mean(), color='blue', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+50, max_ylim*0.9, '2022 Mean: {:.2f}'.format(df_cycle_2022['EP_OutdoorUnit_PowerSum_W'].mean()),color='blue')
    
    plt.axvline(df_cycle_2023['EP_OutdoorUnit_PowerSum_W'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+50, max_ylim*0.8, '2023 Mean: {:.2f}'.format(df_cycle_2023['EP_OutdoorUnit_PowerSum_W'].mean()),color='red')
   
    plt.axvline(df_cycle_2024['EP_OutdoorUnit_PowerSum_W'].mean(), color='green', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+50, max_ylim*0.7, '2024 Mean: {:.2f}'.format(df_cycle_2024['EP_OutdoorUnit_PowerSum_W'].mean()),color='green')

    plt.text(min_xlim+50, max_ylim*0.6, 'Latest Mean: {:.2f}'.format(df_latest['EP_OutdoorUnit_PowerSum_W'].mean()),color='black')

    plt.legend(loc='upper right')
    plt.xlabel('Outdoor Unit Power [W]')
    plt.ylabel('Count')
    plt.title('Space Heating Mode')

heat_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end)

# Water Heating Mode ODU Power
def wh_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end):
    
    df_latest = df_cycle_2024.loc[latest_begin:latest_end]
        
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['Water_Heating_Mode'] == 1]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['Water_Heating_Mode'] == 1]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['Water_Heating_Mode'] == 1]
    df_latest = df_latest[df_latest['Water_Heating_Mode'] == 1]
    
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['AC_and_DHW_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['AC_and_DHW_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['AC_and_DHW_Mode'] == 0]
    df_latest = df_latest[df_latest['AC_and_DHW_Mode'] == 0]
    
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['Defrost_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['Defrost_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['Defrost_Mode'] == 0]
    df_latest = df_latest[df_latest['Defrost_Mode'] == 0]
    
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['Space_Heat_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['Space_Heat_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['Space_Heat_Mode'] == 0]
    df_latest = df_latest[df_latest['Space_Heat_Mode'] == 0]
    
    # Drop cycles from 2022, 2023 data where ODU power < 400W, monitoring issues
    # Do not drop cycles from 2024, to catch new monitoring issues
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['EP_OutdoorUnit_PowerSum_W'] > 400]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['EP_OutdoorUnit_PowerSum_W'] > 400]

    plt.figure()
    if len(df_cycle_2022) > 0:
        plt.hist(df_cycle_2022['EP_OutdoorUnit_PowerSum_W'],alpha=0.5,label='2022',color='blue')
    if len(df_cycle_2023) > 0:
        plt.hist(df_cycle_2023['EP_OutdoorUnit_PowerSum_W'],alpha=0.5,label='2023',color='red')
    if len(df_cycle_2024) > 0:
        plt.hist(df_cycle_2024['EP_OutdoorUnit_PowerSum_W'],alpha=0.5,label='2024',color='green')

    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    plt.axvline(df_cycle_2022['EP_OutdoorUnit_PowerSum_W'].mean(), color='blue', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+50, max_ylim*0.9, '2022 Mean: {:.2f}'.format(df_cycle_2022['EP_OutdoorUnit_PowerSum_W'].mean()),color='blue')
    
    plt.axvline(df_cycle_2023['EP_OutdoorUnit_PowerSum_W'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+50, max_ylim*0.8, '2023 Mean: {:.2f}'.format(df_cycle_2023['EP_OutdoorUnit_PowerSum_W'].mean()),color='red')
   
    plt.axvline(df_cycle_2024['EP_OutdoorUnit_PowerSum_W'].mean(), color='green', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+50, max_ylim*0.7, '2024 Mean: {:.2f}'.format(df_cycle_2024['EP_OutdoorUnit_PowerSum_W'].mean()),color='green')

    plt.text(min_xlim+50, max_ylim*0.6, 'Latest Mean: {:.2f}'.format(df_latest['EP_OutdoorUnit_PowerSum_W'].mean()),color='black')

    plt.legend(loc='upper right')
    plt.xlabel('Outdoor Unit Power [W]')
    plt.ylabel('Count')
    plt.title('Water Heating Mode')

wh_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end)


#%% Histograms of Air Handling Unit (AHU) Power

# HVAC Cooling Mode AHU Power
def cool_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end):
    
    df_latest = df_cycle_2024.loc[latest_begin:latest_end]
        
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['AC_Mode'] == 1]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['AC_Mode'] == 1]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['AC_Mode'] == 1]
    df_latest = df_latest[df_latest['AC_Mode'] == 1]
    
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['AC_and_DHW_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['AC_and_DHW_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['AC_and_DHW_Mode'] == 0]
    df_latest = df_latest[df_latest['AC_and_DHW_Mode'] == 0]

    plt.figure()
    if len(df_cycle_2022) > 0:
        plt.hist(df_cycle_2022['EP_AH_PowerSum_W'],alpha=0.5,label='2022',color='blue')
    if len(df_cycle_2023) > 0:
        plt.hist(df_cycle_2023['EP_AH_PowerSum_W'],alpha=0.5,label='2023',color='red')
    if len(df_cycle_2024) > 0:
        plt.hist(df_cycle_2024['EP_AH_PowerSum_W'],alpha=0.5,label='2024',color='green')

    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    plt.axvline(df_cycle_2022['EP_AH_PowerSum_W'].mean(), color='blue', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+25, max_ylim*0.9, '2022 Mean: {:.2f}'.format(df_cycle_2022['EP_AH_PowerSum_W'].mean()),color='blue')
    
    plt.axvline(df_cycle_2023['EP_AH_PowerSum_W'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+25, max_ylim*0.8, '2023 Mean: {:.2f}'.format(df_cycle_2023['EP_AH_PowerSum_W'].mean()),color='red')
   
    plt.axvline(df_cycle_2024['EP_AH_PowerSum_W'].mean(), color='green', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+25, max_ylim*0.7, '2024 Mean: {:.2f}'.format(df_cycle_2024['EP_AH_PowerSum_W'].mean()),color='green')

    plt.text(min_xlim+25, max_ylim*0.6, 'Latest Mean: {:.2f}'.format(df_latest['EP_AH_PowerSum_W'].mean()),color='black')

    plt.legend(loc='upper right')
    plt.xlabel('Air Handler Power [W]')
    plt.ylabel('Count')
    plt.title('Space Cooling Mode')

cool_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end)

# HVAC Heating Mode AHU Power
def heat_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end):
    
    df_latest = df_cycle_2024.loc[latest_begin:latest_end]
        
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['Space_Heat_Mode'] == 1]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['Space_Heat_Mode'] == 1]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['Space_Heat_Mode'] == 1]
    df_latest = df_latest[df_latest['Space_Heat_Mode'] == 1]
    
    df_cycle_2022 = df_cycle_2022[df_cycle_2022['Defrost_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['Defrost_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['Defrost_Mode'] == 0]
    df_latest = df_latest[df_latest['Defrost_Mode'] == 0]

    plt.figure()
    if len(df_cycle_2022) > 0:
        plt.hist(df_cycle_2022['EP_AH_PowerSum_W'],alpha=0.5,label='2022',color='blue')
    if len(df_cycle_2023) > 0:
        plt.hist(df_cycle_2023['EP_AH_PowerSum_W'],alpha=0.5,label='2023',color='red')
    if len(df_cycle_2024) > 0:
        plt.hist(df_cycle_2024['EP_AH_PowerSum_W'],alpha=0.5,label='2024',color='green')

    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    plt.axvline(df_cycle_2022['EP_AH_PowerSum_W'].mean(), color='blue', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+25, max_ylim*0.9, '2022 Mean: {:.2f}'.format(df_cycle_2022['EP_AH_PowerSum_W'].mean()),color='blue')

    plt.axvline(df_cycle_2023['EP_AH_PowerSum_W'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+25, max_ylim*0.8, '2023 Mean: {:.2f}'.format(df_cycle_2023['EP_AH_PowerSum_W'].mean()),color='red')

    plt.axvline(df_cycle_2024['EP_AH_PowerSum_W'].mean(), color='green', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+25, max_ylim*0.7, '2024 Mean: {:.2f}'.format(df_cycle_2024['EP_AH_PowerSum_W'].mean()),color='green')

    plt.text(min_xlim+25, max_ylim*0.6, 'Latest Mean: {:.2f}'.format(df_latest['EP_AH_PowerSum_W'].mean()),color='black')

    plt.legend(loc='upper right')
    plt.xlabel('Air Handler Power [W]')
    plt.ylabel('Count')
    plt.title('Space Heating Mode')

heat_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end)

# Water Heating Mode AHU Power
def wh_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end):
    
    df_latest = df_cycle_2024.loc[latest_begin:latest_end]

    df_cycle_2022 = df_cycle_2022[df_cycle_2022['Water_Heating_Mode'] == 1]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['Water_Heating_Mode'] == 1]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['Water_Heating_Mode'] == 1]

    df_cycle_2022 = df_cycle_2022[df_cycle_2022['AC_and_DHW_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['AC_and_DHW_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['AC_and_DHW_Mode'] == 0]

    df_cycle_2022 = df_cycle_2022[df_cycle_2022['Defrost_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['Defrost_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['Defrost_Mode'] == 0]

    df_cycle_2022 = df_cycle_2022[df_cycle_2022['Space_Heat_Mode'] == 0]
    df_cycle_2023 = df_cycle_2023[df_cycle_2023['Space_Heat_Mode'] == 0]
    df_cycle_2024 = df_cycle_2024[df_cycle_2024['Space_Heat_Mode'] == 0]

    plt.figure()
    if len(df_cycle_2022) > 0:
        plt.hist(df_cycle_2022['EP_AH_PowerSum_W'],alpha=0.5,label='2022',color='blue')
    if len(df_cycle_2023) > 0:
        plt.hist(df_cycle_2023['EP_AH_PowerSum_W'],alpha=0.5,label='2023',color='red')
    if len(df_cycle_2024) > 0:
        plt.hist(df_cycle_2024['EP_AH_PowerSum_W'],alpha=0.5,label='2024',color='green')

    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    plt.axvline(df_cycle_2022['EP_AH_PowerSum_W'].mean(), color='blue', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+60, max_ylim*0.9, '2022 Mean: {:.2f}'.format(df_cycle_2022['EP_AH_PowerSum_W'].mean()),color='blue')

    plt.axvline(df_cycle_2023['EP_AH_PowerSum_W'].mean(), color='red', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+60, max_ylim*0.8, '2023 Mean: {:.2f}'.format(df_cycle_2023['EP_AH_PowerSum_W'].mean()),color='red')

    plt.axvline(df_cycle_2024['EP_AH_PowerSum_W'].mean(), color='green', linestyle='dashed', linewidth=1)
    plt.text(min_xlim+60, max_ylim*0.7, '2024 Mean: {:.2f}'.format(df_cycle_2024['EP_AH_PowerSum_W'].mean()),color='green')

    plt.text(min_xlim+60, max_ylim*0.6, 'Latest Mean: {:.2f}'.format(df_latest['EP_AH_PowerSum_W'].mean()),color='black')

    plt.legend(loc='upper right')
    plt.xlabel('Air Handler Power [W]')
    plt.ylabel('Count')
    plt.title('Water Heating Mode')

wh_mode_odu_power(df_cycle_2022,df_cycle_2023,df_cycle_2024,latest_begin,latest_end)



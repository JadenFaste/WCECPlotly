import pandas as pd

file_path = "https://raw.githubusercontent.com/JadenFaste/WCECPlotly/main/Test%20data.csv"
df = pd.read_csv(file_path)

heating_column = 'TP_Capacity_Heating_Btuh'
cooling_column = 'TE_Capacity_Cooling_Btu'

# Filter the rows where 'TP_Capacity_Heating_Btuh' is NaN and 'TE_Capacity_Cooling_Btu' is not 0
missing_heating_with_nonzero_cooling = df[(df[heating_column].isna()) & (df[cooling_column] != 0)]

# Count the number of occurrences where 'TE_Capacity_Cooling_Btu' is NOT 0
count_nonzero_cooling = len(missing_heating_with_nonzero_cooling)

# Print the count
print(count_nonzero_cooling)

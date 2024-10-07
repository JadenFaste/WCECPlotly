import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
import os

path = r'C:\Users\bober\OneDrive\Documents\GitHub\WCECPlotly'
os.chdir(path)

# Load and preprocess the data
data = pd.read_csv('cycle_data_updated.csv')
data['idx_Cycle_Start'] = pd.to_datetime(data['idx_Cycle_Start'], errors='coerce')
data['Year'] = data['idx_Cycle_Start'].dt.year

# Filter the data for the first chart
filtered_data_ac_mode = data[(data['AC_Mode'] == 1) & (data['AC_and_DHW_Mode'] == 0)]
means_ac_mode = filtered_data_ac_mode.groupby('Year')['EP_OutdoorUnit_PowerSum_W'].mean().reset_index()

# Create the first histogram
fig_ac_mode = px.histogram(
    filtered_data_ac_mode, 
    x='EP_OutdoorUnit_PowerSum_W', 
    title='Frequency of Outdoor Unit Power Sum by Year (AC Mode On, AC and DHW Mode Off)', 
    labels={'EP_OutdoorUnit_PowerSum_W': 'Outdoor Unit Power Sum (W)', 'count': 'Frequency'}, 
    color='Year',
    nbins=50,
    opacity=0.6
)

y_position = 0.95
y_step = -0.05
for _, row in means_ac_mode.iterrows():
    fig_ac_mode.add_annotation(
        x=0.01,
        y=y_position,
        xref='paper',
        yref='paper',
        text=f"Mean ({int(row['Year'])}): {row['EP_OutdoorUnit_PowerSum_W']:.2f} W",
        showarrow=False,
        font=dict(color="black", size=12),
        align='left'
    )
    y_position += y_step

# Filter the data for the second chart
filtered_data_space_heat = data[(data['Space_Heat_Mode'] == 1) & (data['Defrost_Mode'] == 0)]
means_space_heat = filtered_data_space_heat.groupby('Year')['EP_OutdoorUnit_PowerSum_W'].mean().reset_index()

# Create the second histogram
fig_space_heat = px.histogram(
    filtered_data_space_heat, 
    x='EP_OutdoorUnit_PowerSum_W', 
    title='Frequency of Outdoor Unit Power Sum by Year (Space Heat Mode On, Defrost Mode Off)', 
    labels={'EP_OutdoorUnit_PowerSum_W': 'Outdoor Unit Power Sum (W)', 'count': 'Frequency'}, 
    color='Year',
    nbins=50,
    opacity=0.6
)

y_position = 0.95
for _, row in means_space_heat.iterrows():
    fig_space_heat.add_annotation(
        x=0.01,
        y=y_position,
        xref='paper',
        yref='paper',
        text=f"Mean ({int(row['Year'])}): {row['EP_OutdoorUnit_PowerSum_W']:.2f} W",
        showarrow=False,
        font=dict(color="black", size=12),
        align='left'
    )
    y_position += y_step

# Filter the data for the third chart
filtered_data_water_heating = data[
    (data['Water_Heating_Mode'] == 1) &
    (data['AC_and_DHW_Mode'] == 0) &
    (data['Defrost_Mode'] == 0) &
    (data['Space_Heat_Mode'] == 0) &
    ~((data['Year'].isin([2022, 2023])) & (data['EP_OutdoorUnit_PowerSum_W'] <= 400))
]
means_water_heating = filtered_data_water_heating.groupby('Year')['EP_OutdoorUnit_PowerSum_W'].mean().reset_index()

# Create the third histogram
fig_water_heating = px.histogram(
    filtered_data_water_heating, 
    x='EP_OutdoorUnit_PowerSum_W', 
    title='Frequency of Outdoor Unit Power Sum by Year (Water Heating Mode On, No AC/DHW, Defrost, or Space Heat Mode)', 
    labels={'EP_OutdoorUnit_PowerSum_W': 'Outdoor Unit Power Sum (W)', 'count': 'Frequency'}, 
    color='Year',
    nbins=50,
    opacity=0.6
)

y_position = 0.95
for _, row in means_water_heating.iterrows():
    fig_water_heating.add_annotation(
        x=0.01,
        y=y_position,
        xref='paper',
        yref='paper',
        text=f"Mean ({int(row['Year'])}): {row['EP_OutdoorUnit_PowerSum_W']:.2f} W",
        showarrow=False,
        font=dict(color="black", size=12),
        align='left'
    )
    y_position += y_step

# Filter data for the fourth chart
filtered_data_ah_power = data[(data['AC_Mode'] == 1) & (data['AC_and_DHW_Mode'] == 0)]
means_ah_power = filtered_data_ah_power.groupby('Year')['EP_AH_PowerSum_W'].mean().reset_index()

# Create the fourth histogram
fig_ah_power = px.histogram(
    filtered_data_ah_power, 
    x='EP_AH_PowerSum_W', 
    title='Frequency of Air Handler Power Sum by Year (AC Mode On, AC and DHW Mode Off)', 
    labels={'EP_AH_PowerSum_W': 'Air Handler Power Sum (W)', 'count': 'Frequency'}, 
    color='Year',
    nbins=50,
    opacity=0.6
)

y_position = 0.95
for _, row in means_ah_power.iterrows():
    fig_ah_power.add_annotation(
        x=0.01,
        y=y_position,
        xref='paper',
        yref='paper',
        text=f"Mean ({int(row['Year'])}): {row['EP_AH_PowerSum_W']:.2f} W",
        showarrow=False,
        font=dict(color="black", size=12),
        align='left'
    )
    y_position += y_step

# Filter data for the fifth chart
filtered_data_space_heat_ah = data[(data['Space_Heat_Mode'] == 1) & (data['Defrost_Mode'] == 0)]
means_space_heat_ah = filtered_data_space_heat_ah.groupby('Year')['EP_AH_PowerSum_W'].mean().reset_index()

# Create the fifth histogram
fig_space_heat_ah = px.histogram(
    filtered_data_space_heat_ah, 
    x='EP_AH_PowerSum_W', 
    title='Frequency of Air Handler Power Sum by Year (Space Heat Mode On, Defrost Mode Off)', 
    labels={'EP_AH_PowerSum_W': 'Air Handler Power Sum (W)', 'count': 'Frequency'}, 
    color='Year',
    nbins=50,
    opacity=0.6
)

y_position = 0.95
for _, row in means_space_heat_ah.iterrows():
    fig_space_heat_ah.add_annotation(
        x=0.01,
        y=y_position,
        xref='paper',
        yref='paper',
        text=f"Mean ({int(row['Year'])}): {row['EP_AH_PowerSum_W']:.2f} W",
        showarrow=False,
        font=dict(color="black", size=12),
        align='left'
    )
    y_position += y_step

# Filter data for the sixth chart (new plot)
filtered_data_water_heating_ah = data[
    (data['Water_Heating_Mode'] == 1) &
    (data['AC_and_DHW_Mode'] == 0) &
    (data['Defrost_Mode'] == 0) &
    (data['Space_Heat_Mode'] == 0)
]
means_water_heating_ah = filtered_data_water_heating_ah.groupby('Year')['EP_AH_PowerSum_W'].mean().reset_index()

# Create the sixth histogram
fig_water_heating_ah = px.histogram(
    filtered_data_water_heating_ah, 
    x='EP_AH_PowerSum_W', 
    title='Frequency of Air Handler Power Sum by Year (Water Heating Mode On, No AC/DHW, Defrost, or Space Heat Mode)', 
    labels={'EP_AH_PowerSum_W': 'Air Handler Power Sum (W)', 'count': 'Frequency'}, 
    color='Year',
    nbins=50,
    opacity=0.6
)

y_position = 0.95
for _, row in means_water_heating_ah.iterrows():
    fig_water_heating_ah.add_annotation(
        x=0.01,
        y=y_position,
        xref='paper',
        yref='paper',
        text=f"Mean ({int(row['Year'])}): {row['EP_AH_PowerSum_W']:.2f} W",
        showarrow=False,
        font=dict(color="black", size=12),
        align='left'
    )
    y_position += y_step

# Initialize the Dash app
app = Dash(__name__)
# Define the layout with the five charts stacked vertically
app.layout = html.Div([
    html.H1("AC, Space Heat, Water Heating, and Air Handler Mode Analysis"),
    dcc.Graph(figure=fig_ac_mode),
    dcc.Graph(figure=fig_space_heat),
    dcc.Graph(figure=fig_water_heating),
    dcc.Graph(figure=fig_ah_power),
    dcc.Graph(figure=fig_space_heat_ah),
    dcc.Graph(figure=fig_water_heating_ah)
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)